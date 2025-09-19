import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyMuPDFLoader
from core.user_data import get_user_data
from utils.logging_utils import logging


def get_uploaded_pdf_content(pdf_f_path, username, progress):
    """
    Extract text content from a PDF file.

    Args:
        pdf_f_path: Path to the PDF file
        username: User's username
        progress: Gradio progress object

    Returns:
        Extracted text content
    """
    progress(0, desc="Processing file")
    logging.info(f"Processing PDF {pdf_f_path} for user {username}")
    time.sleep(1)
    loader = PyMuPDFLoader(pdf_f_path)
    docs = loader.load()
    all_texts = [doc.page_content for doc in progress.tqdm(docs, desc="Processing file")]
    final_text = " ".join(all_texts)
    user_data = get_user_data(username)
    user_data.file_data = user_data.encrypt_data(final_text)
    progress(1.0, desc="Processing file")
    return final_text