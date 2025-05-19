# Resume Analysis Chatbot

## Overview

The **Resume Analysis Chatbot** is a Streamlit-based web application that allows users to upload a PDF resume and interactively ask questions about the candidate's qualifications, experience, and skills. It processes the resume using Natural Language Processing (NLP) techniques including text extraction, chunking, embeddings, and retrieval-based question answering.

The chatbot leverages:
- **LangChain** for document processing,
- **FAISS** for vector-based retrieval,
- **Sentence Transformers** for embeddings,
- **Ollama (llama3.1:8b)** as the local language model for generating professional responses.

---

## Features

- 📄 **Resume Upload** – Upload and analyze a PDF resume.
- 💬 **Question Answering** – Ask questions based on the resume content.
- 🧠 **Persistent Chat History** – Maintains conversation state during a session.
- ♻️ **Clear Chat History** – Reset chat history with one click.
- 📚 **Source Document Reference** – Shows source excerpts used to generate answers.
- 🎓 **Professional Responses** – Answers are formal and informed by actual resume content.

---

## Prerequisites

- Python 3.8+
- PDF resume (text-extractable, not scanned images)
- Ollama with the llama3.1:8b model installed and running locally or remotely
- Required Python libraries

---

## Installation

### 1. Clone the Repository

bash
git clone <repository-url>
cd resume-analysis-chatbot

### 2. Set Up a Virtual Environment (optional but recommended)
bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

### 3. Install Dependencies

bash
pip install -r requirements.txt

### 4. Set up Ollama
- Install Ollama: https://ollama.com

- Pull the LLaMA 3.1 model:
bash
ollama pull llama3.1:8b

# Usage
### 1. Run the Application
bash
streamlit run app.py
#This will open the Streamlit interface in your browser.

### 2. Upload a Resume
- Use the sidebar to upload a PDF resume

- The system will extract, chunk, and embed the text

### 3. Ask Questions
Examples:

- "What is the candidate's most recent job?"

- "What skills are listed in the resume?"

### 4. Clear Chat History
Click the Clear Chat History button to reset the conversation

📁 Project Structure

bash

resume-analysis-chatbot/
├── app.py              # Main Streamlit app script
├── requirements.txt    # List of Python dependencies
└── README.md           # Project documentation (this file)


📦 Dependencies
- streamlit – Web UI framework

- langchain, langchain-community – Document parsing and question answering

- faiss-cpu – Vector similarity search

- sentence-transformers – For generating embeddings (e.g., all-MiniLM-L6-v2)

- ollama – For using the LLaMA 3.1 model locally

- PyPDFLoader – For extracting text from PDF resumes

📝 Notes
- The Ollama server must be running before launching the app

- Only one resume can be processed at a time – uploading a new one replaces the old

- Resume should be text-based PDFs (no images or scans)

- Uploaded files are stored temporarily and auto-deleted


📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

yaml
---

Let me know if you also want:

- requirements.txt file generated  
- LICENSE file (MIT template)  
- A basic app.py template to start with
