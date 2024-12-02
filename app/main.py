import os
import sys
import logging

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logging_config import setup_logging
from modules.pdf_parser import extract_text_from_pdf, chunk_text
from modules.database_handler import DatabaseHandler
from modules.query_engine import extract_final_answers
from modules.llm_handler import generate_final_answer
from app.config import (
    CHUNK_SIZE,
    PDF_DIR,
    DATABASE_OPTION,
    OLLAMA_BASE_URL,
    MODEL_NAME,
)

def main():
    setup_logging()
    logging.info("Starting the main function")

    # Initialize database handler
    db_handler = DatabaseHandler(option=DATABASE_OPTION)
    logging.info("DatabaseHandler initialized")

    # Parse and process PDFs
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf") and pdf_file == "sample_document.pdf":
            file_path = os.path.join(PDF_DIR, pdf_file)
            text = extract_text_from_pdf(file_path)
            logging.info(f"Extracted text from {pdf_file}")
            chunks = chunk_text(text, CHUNK_SIZE)
            logging.info(f"Chunked text into {len(chunks)} chunks")

            # Add chunks to the database
            db_handler.add_documents(chunks)
            logging.info(f"Added chunks to the database for {pdf_file}")

    # Test query
    user_query = "What is the capital of UAE?"
    results = db_handler.query(user_query)
    logging.info(f"Query results: {results}")

    # Extract final answers
    final_answers = extract_final_answers(results)
    logging.info(f"Final answers: {final_answers}")

    # Generate final answer using the model
    final_answer = generate_final_answer(
        final_answers, user_query, MODEL_NAME, OLLAMA_BASE_URL
    )
    logging.info(f"Final answer: {final_answer}")
    print("Final answer:", final_answer)

if __name__ == "__main__":
    main()