import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.config import USE_GPU, CHROMADB_PATH, EMBEDDING_MODEL
import pandas as pd
import os



class DatabaseHandler:
    def __init__(self, option="chromadb"):
        """
        Initialize the database handler. Currently supports ChromaDB.
        """
        self.option = option
        self.db = None
        self.collection = None
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda" if USE_GPU else "cpu")

        if option == "chromadb":
            self._setup_chromadb()

    def _setup_chromadb(self):
        """
        Initialize ChromaDB with persistent storage.
        """
        print(f"Initializing ChromaDB with persistence directory: {CHROMADB_PATH}")
        
        # Initialize the ChromaDB client
        self.db = chromadb.PersistentClient(path=CHROMADB_PATH)

        # Create or retrieve the "documents" collection
        self.collection = self.db.get_or_create_collection(name="documents")
        print(f"ChromaDB initialized and collection created at: {CHROMADB_PATH}")

    def add_documents(self, documents):
        """
        Add document chunks to the database and persist the data.
        """
        embeddings = self.embedding_model.encode(documents)
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=[{"chunk_id": i} for i in range(len(documents))],
            embeddings=embeddings
        )
        print(f"Added {len(documents)} documents to ChromaDB.")

    def query(self, query, top_k=5):
        """
        Query the database for similar chunks.
        """
        embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k
        )
        return results


if __name__=="__main__":
    # List all parquet files
    CHROMADB_PATH='data/chrome'
    parquet_files = [f for f in os.listdir(CHROMADB_PATH) if f.endswith('.parquet')]
    print("Parquet Files:", parquet_files)

    # Load and inspect a parquet file
    if parquet_files:
        parquet_file_path = os.path.join(CHROMADB_PATH, parquet_files[0])
        df = pd.read_parquet(parquet_file_path)
        print("Contents of Parquet File:")
        print(df.head())