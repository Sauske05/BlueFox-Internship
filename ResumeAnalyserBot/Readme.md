Resume Analysis Chatbot
Overview
The Resume Analysis Chatbot is a Streamlit-based web application that allows users to upload a PDF resume and ask questions about the candidate's qualifications, experience, or skills. The application processes the resume using natural language processing (NLP) techniques, including document loading, text splitting, embeddings, and a retrieval-based question-answering system powered by LangChain and FAISS. The chatbot leverages the Ollama model (llama3.1:8b) for generating professional responses based on the resume content.
Features

Resume Upload: Upload a PDF resume for analysis.
Question Answering: Ask questions about the resume, and receive answers based on the document's content.
Persistent Chat History: Maintains a conversation history across interactions.
Clear Chat History: Option to reset the chat history.
Source Document Reference: Displays excerpts from the resume used to generate answers.
Professional Responses: Answers are provided in a professional tone with details extracted from the resume.

Prerequisites

Python 3.8+
A compatible environment with the required Python libraries (see Installation section).
Access to the Ollama model (llama3.1:8b) running locally or via a server.
A PDF resume file for analysis.

Installation

Clone the Repository:
git clone <repository-url>
cd resume-analysis-chatbot


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required Python packages using the provided requirements.txt or manually:
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers ollama


Set Up Ollama:

Install Ollama following the instructions at Ollama.
Pull the llama3.1:8b model:ollama pull llama3.1:8b


Ensure the Ollama server is running locally or accessible.



Usage

Run the Application:
streamlit run app.py

This will launch the Streamlit web interface in your default browser.

Upload a Resume:

Use the sidebar to upload a PDF resume.
The application will process the resume and prepare it for question-answering.


Ask Questions:

Enter questions about the resume in the chat input box (e.g., "What is the candidate's most recent job?" or "What skills are listed in the resume?").
The chatbot will respond based on the resume content, displaying the answer and relevant source document excerpts.


Clear Chat History:

Click the "Clear Chat History" button to reset the conversation.



Project Structure

app.py: The main Streamlit application script containing the chatbot logic.
README.md: This file, providing project documentation.
requirements.txt: List of required Python packages (create this based on the dependencies listed above).

Dependencies

streamlit: For the web interface.
langchain and langchain-community: For document processing and retrieval-based question answering.
faiss-cpu: For vector storage and similarity search.
sentence-transformers: For generating embeddings using the all-MiniLM-L6-v2 model.
ollama: For interfacing with the llama3.1:8b language model.
PyPDFLoader: For loading and parsing PDF resumes.

Notes

The application assumes the Ollama server is running locally with the llama3.1:8b model available.
The resume must be in PDF format, and the content should be text-extractable (not scanned images).
The chatbot may not answer questions accurately if the resume lacks the relevant information.
The application uses temporary storage for uploaded files, which are automatically cleaned up.

Limitations

The accuracy of responses depends on the quality of the resume and the capabilities of the llama3.1:8b model.
The application processes one resume at a time; re-uploading a new resume overwrites the previous analysis.
Large resumes may increase processing time due to text splitting and embedding generation.

Future Improvements

Support for multiple resume formats (e.g., DOCX, TXT).
Integration with additional language models or cloud-based APIs.
Enhanced UI with more interactive features, such as resume summary generation.
Support for multilingual resumes using appropriate embeddings and models.

Contributing
Contributions are welcome! Please submit a pull request or open an issue on the repository for suggestions or bug reports.
License
This project is licensed under the MIT License.
