import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.config import USE_GPU, CHROMADB_PATH, EMBEDDING_MODEL

class DatabaseHandler:
    def __init__(self, option="chromadb"):
        """
        Initialize the database handler. Currently supports ChromaDB.
        """
        self.option = option
        self.db = None
        self.collection = None
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL, device="cuda" if USE_GPU else "cpu"
        )

        if option == "chromadb":
            self._setup_chromadb()

    def _setup_chromadb(self):
        """
        Initialize ChromaDB with persistent storage.
        """
        print(f"Initializing ChromaDB with persistence directory: {CHROMADB_PATH}")
        self.db = chromadb.PersistentClient(path=CHROMADB_PATH)

        # Create or retrieve the "documents" collection with correct embedding dimension
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.collection = self.db.get_or_create_collection(
            name="documents", embedding_dim=embedding_dim
        )
        print(
            f"ChromaDB initialized and collection created with embedding dimension: {embedding_dim}"
        )

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
            embeddings=embeddings,
        )
        print(f"Added {len(documents)} documents to ChromaDB.")

    def query(self, query, top_k=5):
        """
        Query the database for similar chunks.
        """
        embedding = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=embedding, n_results=top_k
        )
        return results