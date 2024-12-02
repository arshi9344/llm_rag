import fitz  # PyMuPDF
import pdfplumber
import logging

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Falls back to pdfplumber if PyMuPDF fails.
    """
    logging.info(f"Extracting text from {file_path}")
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        logging.debug(f"Extracted {len(text)} characters from {file_path}")
    except Exception as e:
        logging.error(f"Error with PyMuPDF: {e}, switching to pdfplumber")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """
    Split text into chunks of specified size.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    logging.debug(f"Split text into {len(chunks)} chunks of size {chunk_size}")
    return chunks