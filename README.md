# Secure Cooperate PDF RAG ChatBot

A simple Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, OpenAI embeddings, and a local Chroma vector database. The bot reads documents from `source_documents/`, indexes them, and answers questions by retrieving relevant content.

## Features
- Ingests `.txt`, `.md`, and `.pdf` documents
- Builds a local Chroma vector store
- Uses OpenAI embeddings for semantic search
- Runs an interactive chatbot loop

## Setup
1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file from `.env.example` and add your OpenAI API key:
   ```bash
   copy .env.example .env
   ```
4. Add your API key to `.env`:
   ```text
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage
1. Add documents into `source_documents/`.
2. Build the vector store:
   ```bash
   python rag_chatbot\ingest.py
   ```
3. Run the chatbot:
   ```bash
   python rag_chatbot\app.py
   ```

## Notes
- The vector database is persisted in `rag_chatbot/db/`.
- Re-run `ingest.py` when you add or update source documents.
