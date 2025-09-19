import sys
import os
import asyncio
import tempfile
import shutil

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.user_data import get_user_data
from processors.pdf_handlers import get_uploaded_pdf_content
from utils.helpers import extract_zip_file, process_media, process_image_file
from utils.logging_utils import logging
from generators.report_generator import get_ama_impairment_report
from generators.combiner import get_combine_reports


async def process_file(file_paths, username, progress, generation_cancelled):
    """
    Process multiple files (ZIP, PDF, media, images) and generate combined reports.

    Args:
        file_paths: List of file paths
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event for cancellation

    Yields:
        Combined report text or cancellation message
    """
    results = []
    total_files = len(file_paths)
    logging.info(f"User {username} started processing {total_files} files")
    progress(0.0, desc="Processing file")
    await asyncio.sleep(0)

    all_file_paths = []
    temp_extract_dir = tempfile.mkdtemp()

    for file_path_z in file_paths:
        if file_path_z.endswith(".zip"):
            extracted_files = extract_zip_file(file_path_z, temp_extract_dir)
            all_file_paths.extend(extracted_files)
        else:
            all_file_paths.append(file_path_z)

    user_data = get_user_data(username)

    async def process_single_file(file_path, idx):
        file_progress = 0.1 + (idx / len(all_file_paths)) * 0.7
        progress(file_progress, desc="Processing file")
        logging.info(f"Processing file {file_path} ({idx + 1}/{len(all_file_paths)})")
        if file_path.endswith(".pdf"):
            result = await get_report_from_pdf(file_path, idx, username, progress, generation_cancelled)
        elif file_path.endswith((".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3")):
            result = process_media(file_path)
            result = user_data.encrypt_data(result)
        elif file_path.endswith((".jpg", ".jpeg", ".png")):
            result = process_image_file(file_path)
            result = user_data.encrypt_data(result)
        else:
            result = "Unsupported file format"
        await asyncio.sleep(0)
        return result

    tasks = [process_single_file(file_path, idx) for idx, file_path in enumerate(all_file_paths)]
    results = await asyncio.gather(*tasks)
    progress(0.9, desc="Processing file")

    combined_report = ""
    async for partial_report in get_combine_reports(
            [user_data.decrypt_data(r) if isinstance(r, bytes) else r for r in results], username, progress,
            generation_cancelled):
        if partial_report == "Generation cancelled.":
            progress(1.0, desc="Processing file")
            yield "Generation cancelled."
            return
        combined_report = partial_report
        yield combined_report
        await asyncio.sleep(0.01)

    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    progress(1.0, desc="Processing file")
    await asyncio.sleep(0)
    yield combined_report


async def get_report_from_pdf(pdf_f_path, idx, username, progress, generation_cancelled, vectorstore=None):
    """
    Generate an AMA impairment report from a PDF file.

    Args:
        pdf_f_path: Path to the PDF file
        idx: Index of the file being processed
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event for cancellation
        vectorstore: Chroma vectorstore for AMA content

    Returns:
        Encrypted report text or cancellation message
    """
    from generators.report_generator import process_ama_report_batch
    from langchain_community.document_loaders import PyMuPDFLoader

    progress(0.05, desc="Processing file")
    logging.info(f"Loading PDF {pdf_f_path} for user {username}")
    loader = PyMuPDFLoader(pdf_f_path)
    docs = loader.load()
    await asyncio.sleep(0)

    user_data = get_user_data(username)
    if len(docs) <= 5:
        progress(0.3, desc="Getting report")
        combined_text = " ".join(doc.page_content for doc in docs)
        final_response = await get_ama_impairment_report(combined_text, username, progress, generation_cancelled,
                                                         vectorstore)
        progress(1.0, desc="Getting report")
        await asyncio.sleep(0)
        return user_data.encrypt_data(final_response)

    all_responses = []
    total_batches = (len(docs) + 29) // 30
    for i in progress.tqdm(range(0, len(docs), 30), desc="Getting report"):
        if generation_cancelled.is_set():
            progress(1.0, desc="Processing file")
            logging.info(f"Generation cancelled for PDF {idx + 1}")
            return "Generation cancelled."
        batch_progress = 0.1 + (i / len(docs)) * 0.7
        progress(batch_progress, desc="Getting report")
        pages = [doc.page_content for doc in docs[i:i + 30]]
        batch_result = await process_ama_report_batch(pages, username, progress, generation_cancelled)
        all_responses.append(batch_result)
        await asyncio.sleep(0.05)

    progress(0.9, desc="Getting report")
    final_response = await get_ama_impairment_report(" ".join(user_data.decrypt_data(r) for r in all_responses),
                                                     username, progress, generation_cancelled, vectorstore)
    progress(1.0, desc="Getting report")
    await asyncio.sleep(0)
    return user_data.encrypt_data(final_response)