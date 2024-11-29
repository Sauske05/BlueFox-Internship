from dotenv import load_dotenv


#LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore

from langchain.chains import RetrievalQA


#Loading the .env file
import os
load_dotenv()

#Loading the LLM
api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model_name='llama3-8b-8192', groq_api_key=api_key)

#Testing the LLM
query = 'Tell me about yourself'
prompt = hub.pull('rlm/rag-prompt')
response = llm.invoke(prompt.invoke({'question' : query, 'context' : ''})).content
#print(response)

#Define the embeddings we will be using
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


#load the document and split the document using text splitter
file_path = './resume.pdf'

loader = PyPDFLoader(file_path)
pages = loader.load_and_split()


def split_docs(documents, chunk_size = 200, chunk_overlap = 20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(pages)

#print(len(docs))

#Define Vectore store
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever()


#Define the qa_chain
qa_chain =RetrievalQA.from_chain_type(llm = llm, retriever = retriever)
final_response = qa_chain({'query': 'Tell me about ronak'})

print(final_response['result'])