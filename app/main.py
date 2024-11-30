import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pdf_parser import extract_text_from_pdf, chunk_text
from modules.database_handler import DatabaseHandler
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
            query = "Who is the daughter of Sandeep Raheja?"
            results = db_handler.query(query)
            print("Query results:", results)

if __name__ == "__main__":
    main()
