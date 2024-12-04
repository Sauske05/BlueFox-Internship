import streamlit as st
from dotenv import load_dotenv
import os

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
# Load environment variables
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM
llm = ChatGroq(model_name='llama3-8b-8192', groq_api_key=api_key)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Function to split documents
def split_docs(documents, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Convert text to Document format
    docs = [Document(page_content=doc["text"]) for doc in documents]
    return text_splitter.split_documents(docs)


# Streamlit UI
st.title("RAG-Based Resume Bot")

st.write("Upload a resume, and ask questions about it!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Read the PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the document into chunks
    documents = [{"text": text}]
    docs = split_docs(documents)

    # Create the FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

    # Define the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Input query
    query = st.text_input("Ask something about the uploaded resume:")

    if query:
        with st.spinner("Fetching response..."):
            response = qa_chain({'query': query})
            st.write("### Response:")
            st.write(response['result'])
