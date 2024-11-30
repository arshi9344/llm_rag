# Chatbot Project

This project is designed to create a chatbot for handling queries on technical manuals. It parses PDFs, stores embeddings in ChromaDB, and retrieves relevant chunks using an LLM.

## Features
- Parse legal-style documents with tables and figures.
- Store and retrieve document embeddings using ChromaDB with GPU acceleration.
- Modular codebase for easy customization and switching to FAISS-GPU.
- Optimized for local execution on GPUs.

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the application:
   ```
   python app/main.py
   ```

