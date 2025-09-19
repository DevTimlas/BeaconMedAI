import sys
import os
import uuid
import zipfile
import tempfile
from io import BytesIO
import base64
from moviepy import VideoFileClip
import speech_recognition as sr
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from markdown_pdf import MarkdownPdf, Section
from datetime import date
from openai import OpenAI
from langchain_core.vectorstores import VectorStore
from typing import Set, Optional, Dict

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import logging
from core.models import api_key

openai_client = OpenAI(api_key=api_key)


def generate_uuid():
    """
    Generate a unique UUID string.

    Returns:
        UUID as string
    """
    return str(uuid.uuid4())


def transcribe_wav_to_text_sr(wav_file_path):
    """
    Transcribe WAV audio using Google Speech Recognition.

    Args:
        wav_file_path: Path to the WAV file

    Returns:
        Transcribed text or error message
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_file_path) as source:
            logging.info("Listening to the audio...")
            audio_data = recognizer.record(source)
            logging.info("Recognizing speech...")
            text = recognizer.recognize_google(audio_data)
            logging.info("Transcription complete!")
            return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except FileNotFoundError:
        return "File not found. Please check the path."


def extract_audio_from_video(video_file_path):
    """
    Extract audio from a video file and save it as a temporary WAV file.

    Args:
        video_file_path: Path to the video file

    Returns:
        Path to temporary WAV file or error message
    """
    try:
        video_clip = VideoFileClip(video_file_path)
        audio_clip = video_clip.audio
        temp_audio_file = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
        audio_clip.write_audiofile(temp_audio_file)
        audio_clip.close()
        video_clip.close()
        return temp_audio_file
    except Exception as e:
        logging.error(f"An error occurred while extracting audio: {str(e)}")
        return f"An error occurred while extracting audio: {str(e)}"


def transcribe_audio(audio_file_path):
    """
    Transcribe audio using Groq's Whisper implementation.

    Args:
        audio_file_path: Path to the audio file

    Returns:
        Transcribed text or fallback transcription
    """
    from core.models import groq_client
    try:
        with open(audio_file_path, "rb") as file:
            transcription = groq_client().audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3",
                prompt="""The audio contains a discussion or monologue. Your task is to summarize the content of the audio accurately, 
                            including key points, topics, and any relevant details. The discussion may involve one or more speakers,
                             and the content could range from casual conversation to formal discussions. 
                             Focus on capturing the essence of the discussion, including any important statements, decisions,
                              or actions mentioned. 
                        """,
                response_format="text",
                language="en",
            )
        return transcription
    except Exception as e:
        logging.error(f"An error occurred in Groq transcription: {str(e)}")
        return transcribe_wav_to_text_sr(audio_file_path)


def process_media(file_path):
    """
    Process the uploaded media file (audio or video) and return the transcription.

    Args:
        file_path: Path to the media file

    Returns:
        Transcribed text or error message
    """
    if file_path is None:
        return "Please upload an audio or video file."
    if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        audio_file_path = extract_audio_from_video(file_path)
        if audio_file_path.startswith("An error occurred"):
            return audio_file_path
    else:
        audio_file_path = file_path
    transcription = transcribe_audio(audio_file_path)
    return transcription


def create_pdf_from_markdown(md_text):
    """
    Convert markdown text to a PDF buffer.

    Args:
        md_text: Markdown text to convert

    Returns:
        BytesIO buffer containing PDF
    """
    pdf_generator = MarkdownPdf()
    pdf_generator.add_section(Section(md_text, toc=False))
    pdf_buffer = BytesIO()
    pdf_generator.save(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer


def process_image_file(image_path):
    """
    Process an image file to extract text and summarize content.

    Args:
        image_path: Path to the image file

    Returns:
        Summarized text or error message
    """
    today = date.today().strftime("%d-%m-%Y")
    try:
        img = Image.open(image_path)
        max_size = (1000, 1000)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=100, optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""Extract text and summarize in 200 words max. Include:
        - Key entities as key value pairs (names, numbers, dates) <may be doctor/patient/lawyer etc... 's details)
        - Main context
        - Notable details
        just keep in mind that Today is: {today}"""

        model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            openai_api_key=api_key
        )

        response = model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "low"
                }},
            ])
        ])
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return f"Image processing error: {str(e)[:100]}"


def extract_zip_file(zip_path, extract_dir):
    """
    Extract a ZIP file to a temporary directory.

    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract files to

    Returns:
        List of extracted file paths
    """
    logging.info(f"Extracting ZIP file: {zip_path}")
    os.makedirs(extract_dir, exist_ok=True)
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue
                if os.path.basename(file_info.filename).startswith('.'):
                    logging.info(f"Skipping file starting with a dot: {file_info.filename}")
                    continue
                zip_ref.extract(file_info, extract_dir)
                file_path = os.path.join(extract_dir, file_info.filename)
                if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                    extracted_files.append(file_path)
                else:
                    logging.warning(f"Skipping empty or invalid file: {file_path}")
        logging.info(f"Extracted {len(extracted_files)} valid files from {zip_path}")
        return extracted_files
    except Exception as e:
        logging.error(f"Error extracting ZIP file {zip_path}: {str(e)}")
        return []


def estimate_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    """
    Estimate the number of tokens in a text string.

    Args:
        text: Input text
        model: Model for tokenization

    Returns:
        Number of tokens
    """
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_metadata_filter(
    report_text: str,
    vectorstore: VectorStore,
    max_terms: int = 30,
    min_term_length: int = 3,
    custom_stop_words: Set[str] = {'the', 'and', 'for', 'with', 'this', 'that'},
    similarity_threshold: float = 0.7,
) -> Optional[Dict[str, any]]:
    logging.info("Starting metadata filter generation")
    # Truncate input text to reduce API cost and latency
    max_chars = 10000  # Adjust based on typical report length
    report_text = report_text[:max_chars]
    
    # Extract medical terms using OpenAI
    prompt = f"""
    Extract up to {max_terms} medical terms from the following text. Include only terms related to medical conditions, procedures, medications, or anatomical references. Exclude general words, names, numbers, or non-medical terms. Return the terms as a newline-separated list, each term in lowercase.
    Text: {report_text}
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        terms = response.choices[0].message.content.strip().split('\n')
        from utils.web_utils import clean_term
        terms = [clean_term(term, min_term_length, custom_stop_words) for term in terms if term.strip()]
        terms = [term for term in terms if term][:max_terms]
        logging.debug(f"Extracted {len(terms)} terms: {terms}")
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        terms = []
    
    # Fallback to simple keyword matching
    from utils.web_utils import clean_term
    fallback_terms = [clean_term(word, min_term_length, custom_stop_words)
                      for word in report_text.lower().split()
                      if len(word) >= min_term_length][:max_terms]
    # Merge and deduplicate terms, removing Nones
    all_terms = terms + fallback_terms
    terms = list(set([t for t in all_terms if t is not None and t.strip()]))[:max_terms]
    
    if not terms:
        logging.info("No significant terms extracted")
        return None
    
    # Always include terms in the filter
    filter_dict = {"terms": terms}
    
    # Early check: Skip vectorstore queries if None or invalid; otherwise, add chapters if found
    chapters = set()
    if vectorstore is not None:
        try:
            for term in terms:
                docs = vectorstore.similarity_search(term, k=2)
                for doc in docs:
                    if (hasattr(doc, "metadata") and
                        "chapter" in doc.metadata and
                        doc.metadata.get("score", 1.0) >= similarity_threshold):
                        chapter = doc.metadata["chapter"]
                        if chapter:
                            chapters.add(chapter)
                            logging.debug(f"Term '{term}' matched to chapter '{chapter}'")
        except AttributeError as e:
            # Specific catch for NoneType errors (though we checked above, for safety)
            logging.error(f"Vectorstore invalid (e.g., None): {e}. Skipping chapter matching.")
        except Exception as e:
            # Catch other vectorstore errors
            logging.error(f"Vector store query failed: {e}")
    
    if chapters:
        filter_dict["chapter"] = list(chapters)
        logging.info(f"Combined chapter and term filter: {filter_dict}")
    else:
        logging.warning("No chapters found; using term-based filter only.")
        logging.info(f"Term-based filter: {filter_dict}")
    
    return filter_dict