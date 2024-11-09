import os
import requests
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings  # Import Embeddings base class
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# List of URLs to load documents from
urls = [
    "<https://lilianweng.github.io/posts/2023-06-23-agent/>",
    "<https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/>",
    "<https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/>",
]
# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

"""
# Initialize BERT model and tokenizer for question answering
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Configure text splitter for breaking down documents into manageable chunks
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_documents_into_database(documents_path: str, embedding_model) -> Chroma:
    """
    Loads documents from the specified directory, splits them into chunks,
    and stores them in a Chroma database with the specified embedding model.
    """
    print("Loading and splitting documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Loading documents into Chroma with embeddings")
    # Extract the text content for embedding
    texts = [doc.page_content for doc in documents]
    metadata = [doc.metadata for doc in documents]  # optional if you have metadata

    # Initialize the Chroma database with the embedding model
    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadata,  # remove if not using metadata
    )
    print("Finished loading into Chroma.")
    return db

def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path. Supports PDF and Markdown files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs

def ask_mistral_for_embeddings(texts: List[str], model_name: str, endpoint_url: str) -> List[List[float]]:
    """
    Sends text to a Mistral/LLaMA server for embeddings and retrieves the embeddings.
    """
    url = f"{endpoint_url}/v1/embeddings"
    embeddings = []

    for text in texts:
        payload = {"input": text, "model": model_name}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        embeddings.append(response.json()["data"][0]["embedding"])

    return embeddings

# Define the custom embedding class
class Ollama_Embeddings(Embeddings):
    def __init__(self, model_name: str, endpoint_url: str):
        self.model_name = model_name
        self.endpoint_url = endpoint_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return ask_mistral_for_embeddings(texts, self.model_name, self.endpoint_url)

    def embed_query(self, text: str) -> List[float]:
        return ask_mistral_for_embeddings([text], self.model_name, self.endpoint_url)[0]

def bert_ask_model(question: str, context: str) -> str:
    """
    Uses BERT to answer a question based on the provided context.
    """
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    
    # Check if BERT found a valid answer
    if answer.strip():
        return answer.strip()
    else:
        return "No information available."

def query_documents(db: Chroma, query: str, k=1) -> str:
    """
    Queries the Chroma database to retrieve relevant document chunks based on a query.
    """
    retriever = db.as_retriever(search_type="similarity", k=k)
    relevant_docs = retriever.get_relevant_documents(query)

    if not relevant_docs:
        return "No relevant documents found."

    combined_text = " ".join([doc.page_content for doc in relevant_docs])
    print("Combined Document Text for Query:\n", combined_text)
    return combined_text

# Define the question to be used for both retrieval and answer extraction

embed_model="mistral"
ask_model="mistral"
question = "What is the capital of UAE?"
embedding_model = Ollama_Embeddings(model_name=embed_model, endpoint_url="http://127.0.0.1:11434")

# Load documents into the Chroma vector store database with Mistral embeddings
db = load_documents_into_database("C:\\Users\\USER\\Projects\\llm\\test", embedding_model=embedding_model)

# Retrieve the combined text from relevant documents using the question as the query
combined_text = query_documents(db, question, k=1)

# Use BERT to answer the question based on the retrieved text
if combined_text.strip() and combined_text.lower() != "no relevant documents found.":
    response = bert_ask_model(question, combined_text)
else:
    response = "No relevant information available in the document."

# Print the response from BERT
print("BERT response:", response)
