from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch

# Key learning: SFTTrainer simplifies fine-tuning by abstracting the training loop.
# Step 1: Prepare a small dataset for fine-tuning
# Note: Using a simple list of text for practice; real datasets would be larger.
data = [
    {"text": "Question: What is AI? Answer: AI is the simulation of human intelligence in machines."},
    {"text": "Question: Why use machine learning? Answer: It enables predictive analytics and automation."},
]
dataset = Dataset.from_list(data)

# Load model and tokenizer
# Learning: Reuse Hugging Face models for consistency.
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token (required for some models)
# Note: Found out distilgpt2 doesn't have a pad token by default, so setting it.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Configure training arguments
# Key learning: SFTConfig controls optimizers, schedulers, and training hyperparameters.
training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="epoch",
    evaluation_strategy="no",  # Note: Skipping evaluation for simplicity; would add validation dataset in practice.
    optim="adamw_torch",  # Using AdamW optimizer
    lr_scheduler_type="linear",  # Linear learning rate decay
    max_seq_length=128,  # Limit sequence length for efficiency
)

# Initialize SFTTrainer
# Learning: SFTTrainer handles data collation, training loop, and optimization.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",  # Field in dataset to use for training
    tokenizer=tokenizer,
    args=training_args,
)

# Fine-tune the model
trainer.train()

# Test the fine-tuned model
# Note: Checking if the model learned from the dataset.
input_text = "Question: What is AI?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print("Generated Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# Learning: SFTTrainer abstracts the training loop, making fine-tuning more accessible.