# QueryForge – A RAG Chatbot

QueryForge is a Python-based chatbot that lets users upload PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant text from a document and generate answers based on that context.

## Tech Stack

- **Python** – Core logic
- **Streamlit** – User interface
- **pdfplumber** – PDF text extraction
- **LangChain** – Connecting the retrieval and LLM components
- **Sentence Transformers** – Generating text embeddings locally
- **FAISS** – Storing and retrieving vector embeddings
- **Google Gemini API** – Generating responses

## How It Works

1. The user uploads a PDF.
2. Text is extracted using `pdfplumber`.
3. The extracted text is split into smaller chunks.
4. Each chunk is converted into an embedding using a Sentence Transformers model.
5. The embeddings are stored in a FAISS vector store.
6. When the user asks a question, the retriever finds relevant chunks from the document.
7. The retrieved context and question are passed to Gemini to generate an answer.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vasishta-bhat/QueryForge.git
cd QueryForge
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_api_key_here
```

Get an API key from [Google AI Studio](https://aistudio.google.com/).

Keep the `.env` file out of version control.

### 5. Run the application

```bash
streamlit run chatbot.py
```

Open the local URL provided by Streamlit to use the application.

## Project Structure

```text
QueryForge/
├── chatbot.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Future Improvements

- Support multiple PDFs.
- Display the source passages used to generate answers.
- Add conversation history.
- Handle scanned PDFs.
- Explore local LLMs to reduce reliance on external APIs.

## What I'm Learning

This project is helping me understand how RAG applications work in practice, including document processing, embeddings, vector search, and integrating an LLM into a Python application.
