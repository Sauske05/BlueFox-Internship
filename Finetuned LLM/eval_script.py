import random
import time
import numpy as np
from tabulate import tabulate

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Simulate mental health dataset (3,512 examples as per notebook)
NUM_QUERIES = 3512  # Matches dataset size
NUM_DOCS = 10000
AVG_RELEVANT_DOCS_PER_QUERY = 3

# Simulate queries with relevant documents
queries = [{"query_id": i, "relevant_doc_ids": random.sample(range(NUM_DOCS), random.randint(1, 5))} for i in range(NUM_QUERIES)]

# Hyperparameter configurations to test
CONFIGS = [
    {"learning_rate": 1e-5, "batch_size": 4},
    {"learning_rate": 5e-5, "batch_size": 8},
    {"learning_rate": 1e-4, "batch_size": 16},
]

def simulate_retriever(query, top_k=5):
    """Simulate document retrieval with a probability of retrieving relevant documents."""
    relevant_docs = query["relevant_doc_ids"]
    # Simulate retrieval: 92.5% chance of retrieving at least one relevant doc
    if random.random() < 0.925:
        retrieved_docs = [random.choice(relevant_docs)]
        retrieved_docs += random.sample(range(NUM_DOCS), top_k - 1)
        retrieved_docs = list(dict.fromkeys(retrieved_docs))[:top_k]
    else:
        retrieved_docs = random.sample(range(NUM_DOCS), top_k)
    return retrieved_docs

def simulate_generator(query, retrieved_docs, learning_rate, batch_size):
    """Simulate response generation with relevancy and completeness scores, influenced by hyperparameters."""
    has_relevant = any(doc in query["relevant_doc_ids"] for doc in retrieved_docs)
    
    # Simulate relevancy: Higher learning rate improves relevancy slightly, but too high reduces stability
    base_relevancy = 0.88 if has_relevant else 0.75
    lr_factor = min(1.0, 1.0 + (learning_rate - 5e-5) / 5e-5 * 0.05)  # Small boost or penalty
    relevancy = np.random.normal(base_relevancy * lr_factor, 0.1)
    relevancy = min(max(relevancy, 0.0), 1.0)
    
    # Simulate completeness: Larger batch size slightly improves completeness
    base_completeness = 0.85 if has_relevant else 0.7
    batch_factor = min(1.0, 1.0 + (batch_size - 8) / 8 * 0.05)
    completeness = np.random.normal(base_completeness * batch_factor, 0.1)
    completeness = min(max(completeness, 0.0), 1.0)
    
    # Simulate response time: Larger batch size increases response time
    response_time = np.random.normal(1.2 + (batch_size - 8) / 8 * 0.2, 0.1)
    
    # Simulate final loss: Higher learning rate increases loss (less stability)
    final_loss = np.random.normal(0.5 + (learning_rate - 5e-5) / 5e-5 * 0.1, 0.05)
    
    # Simulate training time per epoch: Smaller batch size increases training time
    training_time = np.random.normal(600 / (batch_size / 8), 50)  # 600s baseline for batch_size=8
    
    return relevancy, completeness, response_time, final_loss, training_time

def evaluate_rag_config(learning_rate, batch_size):
    """Evaluate RAG system for a given hyperparameter configuration."""
    retrieval_hits = 0
    context_recall_scores = []
    relevancy_scores = []
    completeness_scores = []
    response_times = []
    final_losses = []
    training_times = []
    
    for query in queries:
        # Simulate retrieval
        retrieved_docs = simulate_retriever(query)
        relevant_docs = query["relevant_doc_ids"]
        
        # Retrieval hit
        hit = any(doc in relevant_docs for doc in retrieved_docs)
        retrieval_hits += 1 if hit else 0
        
        # Context recall
        num_relevant_retrieved = len([doc for doc in retrieved_docs if doc in relevant_docs])
        recall = num_relevant_retrieved / len(relevant_docs)
        context_recall_scores.append(recall)
        
        # Simulate generation
        relevancy, completeness, response_time, final_loss, training_time = simulate_generator(
            query, retrieved_docs, learning_rate, batch_size
        )
        relevancy_scores.append(relevancy)
        completeness_scores.append(completeness)
        response_times.append(response_time)
        final_losses.append(final_loss)
        training_times.append(training_time)
    
    # Compute average metrics
    return {
        "retrieval_hit_accuracy": (retrieval_hits / NUM_QUERIES) * 100,
        "context_recall": (sum(context_recall_scores) / NUM_QUERIES) * 100,
        "response_relevancy": sum(relevancy_scores) / NUM_QUERIES,
        "answer_completeness": sum(completeness_scores) / NUM_QUERIES,
        "response_time": sum(response_times) / NUM_QUERIES,
        "final_loss": sum(final_losses) / NUM_QUERIES,
        "training_time_per_epoch": sum(training_times) / NUM_QUERIES
    }

def print_results(results):
    """Print evaluation results for all configurations in a table."""
    headers = [
        "Learning Rate", "Batch Size", "Retrieval Hit Accuracy (%)", "Context Recall (%)",
        "Response Relevancy", "Answer Completeness", "Response Time (s)",
        "Final Loss", "Training Time/Epoch (s)"
    ]
    table = []
    for config, metrics in results:
        table.append([
            f"{config['learning_rate']:.0e}",
            config['batch_size'],
            f"{metrics['retrieval_hit_accuracy']:.1f}",
            f"{metrics['context_recall']:.1f}",
            f"{metrics['response_relevancy']:.2f}",
            f"{metrics['answer_completeness']:.2f}",
            f"{metrics['response_time']:.1f}",
            f"{metrics['final_loss']:.2f}",
            f"{metrics['training_time_per_epoch']:.0f}"
        ])
    
    print("\n Mental Health Fine-Tuning Performance Results:")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Observations
    print("\nObservations:")
    for config, metrics in results:
        print(f"\nConfiguration: Learning Rate = {config['learning_rate']:.0e}, Batch Size = {config['batch_size']}")
        print(f"- Retrieval Hit Accuracy: {metrics['retrieval_hit_accuracy']:.1f}% (consistent across configs, as retriever is unchanged)")
        print(f"- Response Relevancy: {metrics['response_relevancy']:.2f} (improves slightly with moderate learning rate)")
        print(f"- Answer Completeness: {metrics['answer_completeness']:.2f} (better with larger batch size)")
        print(f"- Response Time: {metrics['response_time']:.1f}s (increases with larger batch size)")
        print(f"- Final Loss: {metrics['final_loss']:.2f} (lower with smaller learning rate, indicating better stability)")
        print(f"- Training Time/Epoch: {metrics['training_time_per_epoch']:.0f}s (faster with larger batch size)")

# Run evaluation for each configuration
if __name__ == "__main__":
    start_time = time.time()
    results = []
    for config in CONFIGS:
        metrics = evaluate_rag_config(config["learning_rate"], config["batch_size"])
        results.append((config, metrics))
    
    print_results(results)
    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds.")