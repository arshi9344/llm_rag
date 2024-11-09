import os
import requests
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import inquirer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure text splitter for breaking down documents into manageable chunks
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_documents_into_database(model_name: str, documents_path: str, endpoint_url: str) -> Chroma:
    """
    Loads documents from the specified directory, splits them into chunks,
    creates embeddings using the specified model, and stores them in a Chroma database.
    """
    print("Loading and splitting documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    embedding_model = OllamaEmbeddings(model=model_name)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
    )
    print("Finished embedding creation and loading into Chroma.")
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

def ask_model(prompt: str, model_name: str, endpoint_url: str) -> str:
    """
    Sends a prompt to the specified model server and retrieves the generated response.
    """
    url = f"{endpoint_url}/v1/completions"
    payload = {
        "prompt": prompt,
        "model": model_name,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_tokens": 500  # Limit the response length
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with {model_name}: {e}"

def query_documents(db: Chroma, query: str, k=1) -> str:
    """
    Queries the Chroma database to retrieve relevant document chunks based on a query.
    """
    retriever = db.as_retriever(search_type="similarity", k=k)
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return "No relevant documents found."

    combined_text = " ".join([doc.page_content for doc in relevant_docs])
    print("Combined Document Text for Query:\n", combined_text)
    return combined_text

# Endpoint URL configuration
endpoint_url = "http://127.0.0.1:11434"
# embed_model="nomic-embed-text"
embed_model="llama3.1"
# asking_model="llama3.1"
asking_model="llama3.1"
pdf_directory="C:\\Users\\USER\\Projects\\llm\\test"

# Define the list of questions
questions_list = [
    "Who is [Wolfgang Schulz]?",
    "Tell me the capital of UAE. What is the name of the AI technique powered by modern AI chatbot technology built on top of LLM?",
    "Who is the parent of Aarushi Raheja?",
    "What is the capital of UAE?",
    "Is the great wall of china visible from space?",
    "Who is the father of Aarushi Raheja?",
    "Who is the husband of Pooja Raheja?",
    "Who is the mother of Aarushi Raheja?",
    "Who is the spouse of Sandeep?"
]

# Prompt the user to select a question
questions = [
    inquirer.List('question',
                  message="Select a question to ask",
                  choices=questions_list,
                  ),
]
answers = inquirer.prompt(questions)
question = answers['question']

try:
    # Load documents into the Chroma vector store database
    db = load_documents_into_database(embed_model, pdf_directory, endpoint_url)
except Exception as e:
    logging.error(f"Error loading documents: {e}")
    raise

try:
    # Retrieve the combined text from relevant documents using the question as the query
    combined_text = query_documents(db, question, k=4)
except Exception as e:
    logging.error(f"Error querying documents: {e}")
    combined_text = "No relevant documents found."

# Prepare the prompt for the model
query_text = (
    "You are an assistant who answers questions strictly based on the provided Document Text. "
    "Your primary response should be based on the information given in the Document Text. You may need to link information together and make inferences. "
    "For example, if the Document Text states 'Aarushi is the daughter of Sandeep' and 'Pooja is the wife of Sandeep', you should infer that Sandeep and Pooja are the parents of Aarushi. "
    "Similarly, if the Document Text mentions 'John works at Company X' and 'Company X is located in New York', you should infer that John works in New York. "
    "Another example: if the Document Text says 'Alice graduated from Harvard' and 'Harvard is a prestigious university', you should infer that Alice graduated from a prestigious university. "
    "You may add any factual corrections, additional thoughts, or comments you believe are relevant in the Optional Comments section (but should be relevant to the asked question still), "
    "but only if they provide valuable context or corrections.\n\n"
    f"Document Text:\n{combined_text}\n\n"
    f"Question: {question}\n\n"
    "If the answer is not found in the Document Text, respond with 'No information available.' Only say 'No information available' "
    "if there is genuinely no relevant information in the Document Text.\n\n"
    "Please respond using the following XML format:\n\n"
    "<response>\n"
    "    <as-per-documents-provided>\n"
    "        [Your answer based strictly on the Document Text, or 'No information available']\n"
    "    </as-per-documents-provided>\n"
    "    <comments-as-per-external-information>\n"
    "        [Any factual corrections, additional context, or your interpretation, if applicable]\n"
    "    </comments-as-per-external-information>\n"
    "</response>"
)


# Print the prepared query text for debugging
logging.info(f"Prepared query text: {query_text}")

# Prepare the full prompt for the model by filling in the retrieved document text
if combined_text.strip():
    prompt = query_text.format(combined_text=combined_text, question=question)
    response = ask_model(prompt, asking_model, endpoint_url)
else:
    response = "No relevant information available in the document."

# Print the response from the model
print("Response:", response)
