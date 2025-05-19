from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Key learning: LangChain integrates LLMs and vector stores for production workflows.
# Step 1: Prepare sample documents for the vector database
# Note: In a real system, I'd load actual documents, but using a simple text file for practice.
with open("sample_docs.txt", "w") as f:
    f.write("AI is transforming industries.\nMachine learning enables predictive analytics.\nVector databases store embeddings for fast retrieval.")

# Load and split documents
# Learning: TextSplitter chunks large documents for embedding.
loader = TextLoader("sample_docs.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
# Note: Using Hugging Face embeddings for consistency with the LLM.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)

# Set up the LLM pipeline
# Learning: LangChain can wrap Hugging Face models for easy integration.
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Build the RAG pipeline
# Key learning: RetrievalQA combines vector store retrieval with LLM generation.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Simple method to stuff retrieved docs into prompt
    retriever=vector_store.as_retriever(search_kwargs={"k": 2})
)

# Test the pipeline
query = "What is the role of vector databases in AI?"
response = qa_chain.run(query)
print("RAG Response:", response)

# Learning: The vector store retrieves relevant chunks, and the LLM generates a coherent answer.