import os
import sys
import requests 
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logging_config import setup_logging
import logging
from modules.pdf_parser import extract_text_from_pdf, chunk_text
from modules.database_handler import DatabaseHandler
from app.config import CHUNK_SIZE, PDF_DIR, DATABASE_OPTION
import json

# Ollama API setup
OLLAMA_BASE_URL = "http://127.0.0.1:11434" 
MODEL_NAME="llama3.1"
def ask_model(prompt: str, model_name: str, endpoint_url: str) -> str:
    """
    Sends a prompt to the specified model server and retrieves the generated response.
    """
    url = f"{endpoint_url}/v1/completions"
    payload = {
        "prompt": prompt,
        "model": model_name,
        "temperature": 1.2,
        "top_p": 0.9,
        "top_k": 25,
        "max_tokens": 500  # Limit the response length
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with {model_name}: {e}"

def extract_final_answers(results):
    """
    Extracts and formats the final answers from query results.

    :param results: Query results returned by ChromaDB.
    :return: List of relevant answers.
    """
    final_answers = []
    if results and "documents" in results:
        for idx, document in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][idx] if "metadatas" in results else {}
            distance = results["distances"][0][idx] if "distances" in results else None

            answer = {
                "text": document.strip(),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "distance": distance,
            }
            final_answers.append(answer)

    return final_answers


def query_ollama(model_name, prompt):
    """
    Send a prompt to the Ollama local LLaMA server and get the response.
    """
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
    }

    response = requests.post(OLLAMA_BASE_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")


def generate_final_answer(filtered_chunk, user_query):
    """
    Use Ollama to generate a final answer based on the chunk and query.
    """
    context = filtered_chunk["text"]
    prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
    response = query_ollama(MODEL_NAME, prompt)
    return response


def main():
    setup_logging()
    print("Starting the main function")
    
    # Initialize database handler
    db_handler = DatabaseHandler(option=DATABASE_OPTION)
    print("DatabaseHandler initialized")

    # Parse and process PDFs
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, pdf_file)
            text = extract_text_from_pdf(file_path)
            print(f"Extracted text from {pdf_file}")
            chunks = chunk_text(text, CHUNK_SIZE)
            print(f"Chunked text into {len(chunks)} chunks")

            # Add chunks to the database
            db_handler.add_documents(chunks)
            print(f"Added chunks to the database for {pdf_file}")

            # Test query
            query = "Who is the daughter of Sandeep Raheja?"
            results = db_handler.query(query, top_k=3)
            print("Query results:", results)

            # Finalize the answer
            print("\n--- Final Answers ---")
            final_answers = extract_final_answers(results)
            for idx, answer in enumerate(final_answers, start=1):
                print(f"Answer {idx}: {answer}")
            # Step 1: Generate final answer using the filtered chunk
            filtered_chunk = final_answers[0]
            final_answer = generate_final_answer(filtered_chunk, query)

            # Print the final answer
            print("Final Answer:", final_answer)

            ask_model(prompt, asking_model, endpoint_url)





if __name__ == "__main__":
    main()
