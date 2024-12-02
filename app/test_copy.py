import os
import sys
import requests
import json
from app.logging_config import setup_logging
import logging
from modules.pdf_parser import extract_text_from_pdf, chunk_text
from modules.database_handler import DatabaseHandler
from app.config import CHUNK_SIZE, PDF_DIR, DATABASE_OPTION


# Ollama API setup
OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3-7b"  # Replace with the model name in Ollama if different


def main():
    """
    Main function to process documents, store them in a database, query them, and generate final answers.
    """
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
            
            # Step 1: Use the most relevant chunk to generate the final answer
            if final_answers:
                filtered_chunk = final_answers[0]
                final_answer = generate_final_answer(filtered_chunk, query)

                # Print the final answer
                print("\n--- Final Generated Answer ---")
                print("Final Answer:", final_answer)
            else:
                print("No relevant chunks were found for the query.")


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

    :param model_name: Name of the model served by Ollama.
    :param prompt: The text prompt to send to the model.
    :return: Response from the model.
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

    :param filtered_chunk: The most relevant chunk of text retrieved.
    :param user_query: The user query for generating the answer.
    :return: The generated answer from the LLaMA model.
    """
    context = filtered_chunk["text"]
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )
    response = query_ollama(MODEL_NAME, prompt)
    return response


if __name__ == "__main__":
    main()
