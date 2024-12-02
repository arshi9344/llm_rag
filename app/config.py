# Configuration for the application

CHUNK_SIZE = 250
PDF_DIR = "data/pdfs/"
CHROMADB_PATH = "data/chroma/"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
USE_GPU = True  # Enable GPU for embedding generation
DATABASE_OPTION = "chromadb"  # Options: "chromadb" or "faiss"


# Ollama API setup
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.1"