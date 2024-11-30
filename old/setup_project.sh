#!/bin/bash

# Define project name
PROJECT_NAME="chatbot_project"

# Create directory structure
echo "Creating project directories..."
mkdir -p $PROJECT_NAME/{app/{templates},data/{pdfs,md_files,xml_files,embeddings,faiss_index},modules,tests}

# Create basic files
echo "Creating basic files..."
touch $PROJECT_NAME/{README.md,requirements.txt,setup.py}
touch $PROJECT_NAME/app/{__init__.py,main.py,config.py,logging_config.py}
touch $PROJECT_NAME/app/templates/{default_prompt.yaml,clause_query_template.yaml}
touch $PROJECT_NAME/modules/{__init__.py,pdf_parser.py,data_preprocessor.py,database_handler.py,query_engine.py,llm_handler.py,utils.py}
touch $PROJECT_NAME/tests/{test_pdf_parser.py,test_query_engine.py,test_llm_handler.py}

# Populate README.md
cat <<EOL > $PROJECT_NAME/README.md
# Chatbot Project

This project is designed to create a chatbot for handling queries on technical manuals. It parses PDFs, stores embeddings in ChromaDB, and retrieves relevant chunks using an LLM.

## Features
- Parse legal-style documents with tables and figures.
- Store and retrieve document embeddings using ChromaDB with GPU acceleration.
- Modular codebase for easy customization and switching to FAISS-GPU.
- Optimized for local execution on GPUs.

## How to Run
1. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
2. Run the application:
   \`\`\`
   python app/main.py
   \`\`\`

EOL

# Populate requirements.txt
# cat <<EOL > $PROJECT_NAME/requirements.txt
# flask
# requests
# numpy
# pandas
# scikit-learn
# langchain
# torch
# tqdm
# streamlit
# inquirer
# transformers
# pypdf2
# chromadb
# sentence-transformers
# pdfplumber
# pymupdf
# unstructured
# bitsandbytes
# pyyaml
# accelerate
# uvicorn
# rich
# EOL

# Populate app/config.py
cat <<EOL > $PROJECT_NAME/app/config.py
# Configuration for the application

CHUNK_SIZE = 500
PDF_DIR = "data/pdfs/"
CHROMADB_PATH = "data/chroma/"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
USE_GPU = True  # Enable GPU for embedding generation
DATABASE_OPTION = "chromadb"  # Options: "chromadb" or "faiss"
EOL

# Populate app/logging_config.py
cat <<EOL > $PROJECT_NAME/app/logging_config.py
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
EOL

# Populate app/templates/default_prompt.yaml
cat <<EOL > $PROJECT_NAME/app/templates/default_prompt.yaml
template_name: "default"
prompt: |
  Please provide recommendations based on the following clauses:
  {context}
  Question: {query}
EOL

# Populate app/templates/clause_query_template.yaml
cat <<EOL > $PROJECT_NAME/app/templates/clause_query_template.yaml
template_name: "clause_query"
prompt: |
  Retrieve all relevant clauses for the following question:
  {query}
EOL

# Populate app/main.py
cat <<EOL > $PROJECT_NAME/app/main.py
import os
from app.modules.pdf_parser import extract_text_from_pdf, chunk_text
from app.modules.database_handler import DatabaseHandler
from app.config import CHUNK_SIZE, PDF_DIR, DATABASE_OPTION
from app.logging_config import setup_logging

def main():
    setup_logging()
    # Initialize database handler
    db_handler = DatabaseHandler(option=DATABASE_OPTION)

    # Parse and process PDFs
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, pdf_file)
            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text, CHUNK_SIZE)

            # Add chunks to the database
            db_handler.add_documents(chunks)

            # Test query
            query = "What are the recommendations of API 610 on hydraulic selection of pumps?"
            results = db_handler.query(query)
            print("Query results:", results)

if __name__ == "__main__":
    main()
EOL

# Populate modules/database_handler.py
cat <<EOL > $PROJECT_NAME/modules/database_handler.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

class DatabaseHandler:
    def __init__(self, option="chromadb"):
        self.option = option
        self.db = None
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if USE_GPU else "cpu")
        
        if option == "chromadb":
            self._setup_chromadb()

    def _setup_chromadb(self):
        """
        Initialize ChromaDB with persistent storage.
        """
        logging.info("Initializing ChromaDB...")
        self.db = chromadb.Client(Settings(persist_directory="data/chroma/"))

    def add_documents(self, documents):
        """
        Add document chunks to the database.
        """
        embeddings = self.embedding_model.encode(documents)
        for i, doc in enumerate(documents):
            self.db.add(
                collection_name="documents",
                documents=[doc],
                metadatas=[{"chunk_id": i}],
                embeddings=[embeddings[i]]
            )
        logging.info(f"Added {len(documents)} documents to ChromaDB.")

    def query(self, query, top_k=5):
        """
        Query the database for similar chunks.
        """
        embedding = self.embedding_model.encode([query])
        results = self.db.query(
            collection_name="documents",
            query_embeddings=embedding,
            n_results=top_k
        )
        return results

EOL

# Create empty module files
for file in pdf_parser.py data_preprocessor.py query_engine.py llm_handler.py utils.py; do
    cat <<EOL > $PROJECT_NAME/modules/$file
# TODO: Implement $file module
EOL
done

# Populate tests
for file in test_pdf_parser.py test_query_engine.py test_llm_handler.py; do
    cat <<EOL > $PROJECT_NAME/tests/$file
import unittest

class Test$(basename $file .py | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)}1' | tr -d ' ')(unittest.TestCase):
    def test_placeholder(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
EOL
done

echo "Project setup with ChromaDB complete!"
