import sys
import os
import asyncio
import threading
import tempfile
import markdown
import shutil
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, Response, Cookie
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from auth.authentication import validate_login
from core.user_data import get_user_data, save_chat_history, download_chat_history
from core.vectorstore import initialize_vectorstore_for_edition, current_vectorstore
from processors.file_processors import process_file
from processors.text_processors import check_moderation
from processors.pdf_handlers import get_uploaded_pdf_content
from generators.report_generator import get_ama_impairment_report
from generators.rebuttal_generator import process_batch_for_rebuttal
from generators.combiner import get_combine_reports
from utils.helpers import create_pdf_from_markdown
from utils.logging_utils import logging
from core.models import init_client
import io
import base64
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import timeout_decorator
import concurrent.futures
from functools import partial
import traceback
import hashlib
import pickle
import uuid

app = FastAPI()

# Custom file-based session management (no Redis)
SESSION_DIR = os.path.join(os.getcwd(), 'app_sessions')
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR, mode=0o755, exist_ok=True)

class SimpleSessionManager:
    """Custom session manager that works reliably with multiple workers."""
    def __init__(self, session_dir):
        self.session_dir = session_dir

    def _get_session_file(self, session_id):
        return os.path.join(self.session_dir, f"session_{session_id}.pkl")

    def _generate_session_id(self):
        return str(uuid.uuid4()).replace('-', '')

    def get_session_data(self, session_id):
        if not session_id:
            return {}, None
        session_file = self._get_session_file(session_id)
        try:
            if os.path.exists(session_file):
                # Check if session file is not too old (24 hours)
                file_age = datetime.now().timestamp() - os.path.getmtime(session_file)
                if file_age > 24 * 3600:
                    os.remove(session_file)
                    return {}, None
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                return session_data, session_id
        except Exception as e:
            logging.warning(f"Error loading session {session_id}: {str(e)}")
            return {}, None

    def save_session_data(self, session_data, session_id=None):
        if not session_id:
            session_id = self._generate_session_id()
        session_file = self._get_session_file(session_id)
        try:
            # Add timestamp to session data
            session_data['_last_updated'] = datetime.now().timestamp()
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
            logging.info(f"Session saved: {session_id}")
            return session_id
        except Exception as e:
            logging.error(f"Error saving session {session_id}: {str(e)}")
            return None

    def clear_session(self, session_id):
        if not session_id:
            return
        session_file = self._get_session_file(session_id)
        try:
            if os.path.exists(session_file):
                os.remove(session_file)
                logging.info(f"Session cleared: {session_id}")
        except Exception as e:
            logging.error(f"Error clearing session {session_id}: {str(e)}")

    def cleanup_old_sessions(self):
        try:
            current_time = datetime.now().timestamp()
            for filename in os.listdir(self.session_dir):
                if filename.startswith('session_') and filename.endswith('.pkl'):
                    filepath = os.path.join(self.session_dir, filename)
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > 24 * 3600:
                        os.remove(filepath)
        except Exception as e:
            logging.error(f"Error cleaning up sessions: {str(e)}")

session_manager = SimpleSessionManager(SESSION_DIR)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'.zip', '.pdf', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.jpg', '.jpeg', '.png'}

# Global variables
AUTH_CREDENTIALS = {
    "michael rice": "4321",
    "Tim": "1038",
    "Tester": "Test012"
}

worker_state = threading.local()

def get_generation_running():
    if not hasattr(worker_state, 'generation_running'):
        worker_state.generation_running = threading.Event()
    return worker_state.generation_running

def get_generation_cancelled():
    if not hasattr(worker_state, 'generation_cancelled'):
        worker_state.generation_cancelled = threading.Event()
    return worker_state.generation_cancelled

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def get_session_info(session_id):
    worker_pid = os.getpid()
    session_data, _ = session_manager.get_session_data(session_id)
    username = session_data.get('username', 'None')
    return worker_pid, session_id or 'None', username

def get_session_id(beacon_session_id: str = Cookie(None)):
    return beacon_session_id

def require_login(session_id: str = Depends(get_session_id)):
    session_data, _ = session_manager.get_session_data(session_id)
    if not session_data.get('logged_in') or not session_data.get('username'):
        raise HTTPException(status_code=401, detail='Please log in first.')
    return session_data

def check_text_similarity(text1, text2, method='cosine'):
    try:
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        if not text1 or not text2:
            return 0.0
        text1 = text1.lower()
        text2 = text2.lower()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity, 4)
    except Exception as e:
        logging.error(f"Error in text similarity check: {str(e)}")
        return 0.0

def rerank_documents(query, documents, top_k=2):
    try:
        embedder = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        query_embedding = embedder.embed_query(query)
        doc_embeddings = embedder.embed_documents([doc.page_content for doc in documents])
        similarities = [
            np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in doc_embeddings
        ]
        doc_score_pairs = list(zip(documents, similarities))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair[0] for pair in doc_score_pairs[:top_k]]
    except Exception as e:
        logging.error(f"Error in document reranking: {str(e)}")
        return documents[:top_k]

class DummyProgress:
    def __call__(self, value=None, desc=None, total=None, unit=None, track_tqdm=False):
        if desc:
            logging.info(f"Progress: {desc} ({value}/{total} {unit})")
        return self

    def tqdm(self, iterable, desc=None, total=None):
        logging.info(f"Processing {desc} with {len(iterable) if total is None else total} items")
        return iterable

async def safe_file_cleanup(file_paths):
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {str(e)}")

async def collect_file_results(file_paths, username, progress, cancelled):
    results = []
    try:
        if not file_paths:
            logging.error("No file paths provided to process_file")
            return ["Error: No files provided"]
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logging.error(f"File does not exist: {file_path}")
                return [f"Error: File {file_path} does not exist"]
            try:
                file_size = os.path.getsize(file_path) / 1024
                logging.info(f"Processing file: {file_path}, size: {file_size:.2f} KB")
            except Exception as e:
                logging.error(f"Error accessing file {file_path}: {str(e)}")
                return [f"Error: Cannot access file {file_path}"]
        result = process_file(file_paths, username, progress or DummyProgress(), cancelled)
        if hasattr(result, '__aiter__'):
            async for item in result:
                if cancelled.is_set():
                    logging.info("File processing cancelled by user")
                    return ["Generation cancelled."]
                if item is None:
                    continue
                results.append(str(item))
        else:
            results = [result] if isinstance(result, str) else []
        return results
    except Exception as e:
        logging.error(f"Error in collect_file_results: {str(e)}")
        return [f"Error: {str(e)}"]

async def collect_combine_results(reports, username, progress, cancelled):
    results = []
    try:
        if not reports:
            logging.error("No reports provided to get_combine_reports")
            return ["Error: No reports provided"]
        result = get_combine_reports(reports, username, progress or DummyProgress(), cancelled)
        if hasattr(result, '__aiter__'):
            async for item in result:
                if cancelled.is_set():
                    return ["Generation cancelled."]
                if item is None:
                    continue
                results.append(str(item))
        else:
            results = [result] if isinstance(result, str) else []
        return results
    except Exception as e:
        logging.error(f"Error in collect_combine_results: {str(e)}")
        return [f"Error: {str(e)}"]

@app.on_event("startup")
async def startup_event():
    initialize_vectorstore_for_edition("4th Edition")
    session_manager.cleanup_old_sessions()

@app.get("/")
async def serve_index():
    return FileResponse('index.html')

class LoginData(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def login(data: LoginData, response: Response):
    username = data.username
    password = data.password
    if username in AUTH_CREDENTIALS and AUTH_CREDENTIALS[username] == password and validate_login(username, password):
        session_data = {
            'username': username,
            'logged_in': True,
            'login_time': datetime.now().isoformat()
        }
        session_id = session_manager.save_session_data(session_data)
        if session_id:
            logging.info(f"User {username} logged in - Session ID: {session_id}")
            response.set_cookie(key='beacon_session_id', value=session_id, max_age=24*3600, httponly=True)
            return {'status': 'success', 'username': username}
        else:
            raise HTTPException(status_code=500, detail='Failed to create session')
    logging.error(f"Failed login attempt for username: {username}")
    raise HTTPException(status_code=401, detail='Invalid username or password')

@app.post("/api/logout")
async def logout(response: Response, session_id: str = Depends(get_session_id)):
    session_data, _ = session_manager.get_session_data(session_id)
    username = session_data.get('username')
    if session_id:
        session_manager.clear_session(session_id)
    logging.info(f"User {username} logged out - Session ID: {session_id}")
    response.delete_cookie('beacon_session_id')
    return {'status': 'success', 'message': 'Logged out'}

@app.get("/api/get_username")
async def get_username(session_data: dict = Depends(require_login), session_id: str = Depends(get_session_id)):
    username = session_data.get('username')
    logged_in = session_data.get('logged_in', False)
    logging.info(f"Checking session - User: {username}, Logged in: {logged_in}, Session ID: {session_id}")
    if username and logged_in:
        session_data['last_accessed'] = datetime.now().isoformat()
        session_manager.save_session_data(session_data, session_id)
        return {'username': username}
    return {'username': None}

@app.get("/api/debug_session")
async def debug_session(session_id: str = Depends(get_session_id)):
    worker_pid, _, username = get_session_info(session_id)
    session_data, _ = session_manager.get_session_data(session_id)
    debug_info = {
        'worker_pid': worker_pid,
        'session_id': session_id,
        'username': username,
        'session_data': session_data,
        'session_dir': SESSION_DIR,
        'session_files_exist': os.path.exists(SESSION_DIR),
        'session_file_count': 0,
        'session_files': []
    }
    if os.path.exists(SESSION_DIR):
        try:
            session_files = [f for f in os.listdir(SESSION_DIR) if f.startswith('session_')]
            debug_info['session_file_count'] = len(session_files)
            debug_info['session_files'] = session_files[:10]
        except Exception as e:
            debug_info['session_dir_error'] = str(e)
    logging.info(f"Debug session info: {debug_info}")
    return debug_info

class TestSessionData(BaseModel):
    test_data: str = 'test_value'

@app.post("/api/test_session_write")
async def test_session_write(data: TestSessionData, session_id: str = Depends(get_session_id)):
    test_data = data.test_data
    session_data, _ = session_manager.get_session_data(session_id)
    worker_pid = os.getpid()
    session_data['test_key'] = test_data
    session_data['test_timestamp'] = datetime.now().isoformat()
    saved_session_id = session_manager.save_session_data(session_data, session_id)
    logging.info(f"Set test session data - Session ID: {saved_session_id}")
    return {
        'status': 'success',
        'worker_pid': worker_pid,
        'session_id': saved_session_id,
        'test_data_set': test_data,
        'session_data': session_data
    }

class UpdateVectorstoreData(BaseModel):
    edition: str = '4th Edition'

@app.post("/api/update_vectorstore")
async def update_vectorstore(data: UpdateVectorstoreData, session_data: dict = Depends(require_login)):
    edition = data.edition
    initialize_vectorstore_for_edition(edition)
    logging.info(f"Vectorstore updated to {edition}")
    return {'status': f"Using AMA Guides {edition}"}

@app.post("/api/generate_report", response_class=StreamingResponse)
async def generate_report(
    files: list[UploadFile] = File(None),
    text: str = Form(''),
    edition: str = Form('4th Edition'),
    session_data: dict = Depends(require_login),
    session_id: str = Depends(get_session_id)
):
    username = session_data['username']
    worker_pid, _, _ = get_session_info(session_id)
    logging.info(f"Worker {worker_pid}: Generating report for user: {username}, Session ID: {session_id}")
    logging.info(f"Received {len(files) if files else 0} files for processing")
    for file in files or []:
        if file.filename:
            logging.info(f"File: {file.filename}, type: {file.content_type}")
    initialize_vectorstore_for_edition(edition)
    user_data = get_user_data(username)
    if not files and not text.strip():
        raise HTTPException(status_code=400, detail='Please upload at least one document or provide text input.')

    GENERATION_RUNNING = get_generation_running()
    GENERATION_CANCELLED = get_generation_cancelled()
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()

    file_paths = []
    temp_files_to_cleanup = []
    try:
        # Handle file uploads upfront
        if files:
            for file in files:
                if file.filename and allowed_file(file.filename):
                    logging.info(f"File closed before read: {file.file.closed}")
                    await file.seek(0)
                    contents = await file.read()
                    if len(contents) > 0:
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(UPLOAD_FOLDER, f"{username}_{filename}")
                        with open(file_path, 'wb') as f:
                            f.write(contents)
                        temp_files_to_cleanup.append(file_path)
                        file_paths.append(file_path)
                        file_size = len(contents) / 1024
                        logging.info(f"Successfully uploaded: {filename} ({file_size:.2f} KB)")
                    else:
                        raise HTTPException(status_code=400, detail='Empty file uploaded')
                else:
                    raise HTTPException(status_code=400, detail=f'Invalid file type: {file.filename}')
    except Exception as e:
        await safe_file_cleanup(temp_files_to_cleanup)
        raise HTTPException(status_code=500, detail=f'Error uploading files: {str(e)}')

    async def generate():
        all_reports = []
        try:
            # Process uploaded files
            if file_paths:
                logging.info(f"Processing {len(file_paths)} files")
                partial_results = await collect_file_results(file_paths, username, DummyProgress(), GENERATION_CANCELLED)
                for partial_result in partial_results:
                    if GENERATION_CANCELLED.is_set():
                        yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                        return
                    if partial_result == "Generation cancelled." or partial_result.startswith("Error:"):
                        yield f"data: {json.dumps({'report': partial_result})}\n\n"
                        return
                    all_reports.append(partial_result)
                    try:
                        markdown_report = markdown.markdown(partial_result)
                        logging.info(f"Streaming partial report: {partial_result[:50]}...")
                        yield f"data: {json.dumps({'report': markdown_report})}\n\n"
                    except Exception as e:
                        logging.error(f"Error converting to markdown: {str(e)}")
                        yield f"data: {json.dumps({'report': partial_result})}\n\n"
            # Handle text input
            if text.strip():
                moderation_result = check_moderation(text)
                if moderation_result:
                    yield f"data: {json.dumps({'report': f'Content flagged: {moderation_result}'})}\n\n"
                    return
                logging.info("Processing text input for AMA report")
                text_report = await get_ama_impairment_report(text, username, DummyProgress(), GENERATION_CANCELLED, current_vectorstore)
                if GENERATION_CANCELLED.is_set():
                    yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                    return
                if text_report and not text_report.startswith("Error:"):
                    if not files:
                        logging.info("Combining text report")
                        partial_results = await collect_combine_results([text_report], username, DummyProgress(), GENERATION_CANCELLED)
                        for combined_report in partial_results:
                            if GENERATION_CANCELLED.is_set():
                                yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                                return
                            if combined_report == "Generation cancelled." or combined_report.startswith("Error:"):
                                yield f"data: {json.dumps({'report': combined_report})}\n\n"
                                return
                            all_reports.append(combined_report)
                            try:
                                markdown_report = markdown.markdown(combined_report)
                                logging.info(f"Streaming combined report: {combined_report[:50]}...")
                                yield f"data: {json.dumps({'report': markdown_report})}\n\n"
                            except Exception as e:
                                logging.error(f"Error converting combined report to markdown: {str(e)}")
                                yield f"data: {json.dumps({'report': combined_report})}\n\n"
                    else:
                        all_reports.append(text_report)
                else:
                    error_msg = text_report if text_report else "Failed to generate text report"
                    yield f"data: {json.dumps({'report': error_msg})}\n\n"
                    return
            # Final processing
            if not all_reports:
                yield f"data: {json.dumps({'report': 'No reports generated.'})}\n\n"
                return
            final_report_text = all_reports[-1]
            try:
                user_data.report_data = user_data.encrypt_data(final_report_text)
                logging.info("Report data encrypted and saved")
                pdf_base64 = None
                try:
                    pdf_buffer = create_pdf_from_markdown(final_report_text)
                    if pdf_buffer:
                        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                        logging.info("PDF generated successfully")
                except Exception as pdf_error:
                    logging.error(f"Error creating PDF: {str(pdf_error)}")
                final_response = {'report': markdown.markdown(final_report_text)}
                if pdf_base64:
                    final_response['pdf'] = pdf_base64
                logging.info(f"Streaming final report: {final_report_text[:50]}...")
                yield f"data: {json.dumps(final_response)}\n\n"
            except Exception as e:
                logging.error(f"Error in final processing: {str(e)}")
                yield f"data: {json.dumps({'report': f'Error in final processing: {str(e)}'})}\n\n"
        except Exception as e:
            logging.error(f"Error in report generation: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps({'report': f'Error: {str(e)}'})}\n\n"
        finally:
            await safe_file_cleanup(temp_files_to_cleanup)
            GENERATION_RUNNING.clear()

    return StreamingResponse(generate(), media_type='text/event-stream')

@app.post("/api/generate_rebuttal", response_class=StreamingResponse)
async def generate_rebuttal(
    files: list[UploadFile] = File(None),
    text: str = Form(''),
    edition: str = Form('4th Edition'),
    session_data: dict = Depends(require_login),
    session_id: str = Depends(get_session_id)
):
    username = session_data['username']
    worker_pid = os.getpid()
    logging.info(f"Worker {worker_pid}: Generating rebuttal for user: {username}, Session ID: {session_id}")
    initialize_vectorstore_for_edition(edition)
    user_data = get_user_data(username)
    if not files and not text.strip():
        raise HTTPException(status_code=400, detail='Please upload a document or provide text.')

    GENERATION_RUNNING = get_generation_running()
    GENERATION_CANCELLED = get_generation_cancelled()
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()

    file_paths = []
    temp_files_to_cleanup = []
    try:
        if files:
            for file in files:
                if file.filename and allowed_file(file.filename):
                    logging.info(f"File closed before read: {file.file.closed}")
                    await file.seek(0)
                    contents = await file.read()
                    if len(contents) > 0:
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(UPLOAD_FOLDER, f"{username}_{filename}")
                        with open(file_path, 'wb') as f:
                            f.write(contents)
                        temp_files_to_cleanup.append(file_path)
                        file_paths.append(file_path)
                        file_size = len(contents) / 1024
                        logging.info(f"Successfully uploaded rebuttal file: {filename} ({file_size:.2f} KB")
                    else:
                        raise HTTPException(status_code=400, detail='Empty file uploaded')
                else:
                    raise HTTPException(status_code=400, detail=f'Invalid file type: {file.filename}')
    except Exception as e:
        await safe_file_cleanup(temp_files_to_cleanup)
        raise HTTPException(status_code=500, detail=f'Error uploading files: {str(e)}')

    async def generate():
        all_responses = []
        try:
            if file_paths:
                partial_results = await collect_file_results(file_paths, username, DummyProgress(), GENERATION_CANCELLED)
                for rebuttal_text in partial_results:
                    if GENERATION_CANCELLED.is_set():
                        yield f"data: {json.dumps({'rebuttal': 'Generation cancelled.'})}\n\n"
                        return
                    if rebuttal_text == "Generation cancelled." or rebuttal_text.startswith("Error:"):
                        yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"
                        return
                    all_responses.append(rebuttal_text)
                    try:
                        markdown_rebuttal = markdown.markdown(rebuttal_text)
                        yield f"data: {json.dumps({'rebuttal': markdown_rebuttal})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"
            if text.strip():
                moderation_result = check_moderation(text)
                if moderation_result:
                    yield f"data: {json.dumps({'rebuttal': f'Content flagged: {moderation_result}'})}\n\n"
                    return
                rebuttal_text = await process_batch_for_rebuttal([text], username, DummyProgress(), GENERATION_CANCELLED, current_vectorstore)
                if GENERATION_CANCELLED.is_set():
                    yield f"data: {json.dumps({'rebuttal': 'Generation cancelled.'})}\n\n"
                    return
                if rebuttal_text and rebuttal_text != "Generation cancelled." and not rebuttal_text.startswith("Error:"):
                    all_responses.append(rebuttal_text)
                    try:
                        markdown_rebuttal = markdown.markdown(rebuttal_text)
                        yield f"data: {json.dumps({'rebuttal': markdown_rebuttal})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"
            if not all_responses:
                yield f"data: {json.dumps({'rebuttal': 'No rebuttal generated.'})}\n\n"
                return
            final_rebuttal_text = all_responses[-1]
            try:
                user_data.rebuttal_data = user_data.encrypt_data(final_rebuttal_text)
                logging.info("Rebuttal data encrypted and saved")
                pdf_base64 = None
                try:
                    pdf_buffer = create_pdf_from_markdown(final_rebuttal_text)
                    if pdf_buffer:
                        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                except Exception as pdf_error:
                    logging.error(f"Error creating rebuttal PDF: {str(pdf_error)}")
                final_response = {'rebuttal': markdown.markdown(final_rebuttal_text)}
                if pdf_base64:
                    final_response['pdf'] = pdf_base64
                yield f"data: {json.dumps(final_response)}\n\n"
            except Exception as e:
                logging.error(f"Error in final rebuttal processing: {str(e)}")
                yield f"data: {json.dumps({'rebuttal': f'Error in final processing: {str(e)}'})}\n\n"
        except Exception as e:
            logging.error(f"Error in rebuttal generation: {str(e)}")
            yield f"data: {json.dumps({'rebuttal': f'Error: {str(e)}'})}\n\n"
        finally:
            await safe_file_cleanup(temp_files_to_cleanup)
            GENERATION_RUNNING.clear()

    return StreamingResponse(generate(), media_type='text/event-stream')

@app.post("/api/reset_reports")
async def reset_reports(session_data: dict = Depends(require_login), session_id: str = Depends(get_session_id)):
    username = session_data['username']
    worker_pid = os.getpid()
    user_data = get_user_data(username)
    user_data.file_data = b""
    user_data.report_data = b""
    user_data.rebuttal_data = b""
    logging.info(f"Worker {worker_pid}: Reports reset for user: {username}, Session ID: {session_id}")
    return {'status': 'Reports reset'}

@app.post("/api/process_chat_files")
async def process_chat_files(
    files: list[UploadFile] = File(None),
    session_data: dict = Depends(require_login),
    session_id: str = Depends(get_session_id)
):
    username = session_data['username']
    worker_pid = os.getpid()
    logging.info(f"Worker {worker_pid}: Processing chat files for user: {username}, Session ID: {session_id}")
    temp_files_to_cleanup = []
    try:
        user_data = get_user_data(username)
        all_pdf_content = []
        for file in files or []:
            if file.filename.lower().endswith('.pdf'):
                logging.info(f"File closed before read: {file.file.closed}")
                await file.seek(0)
                contents = await file.read()
                if len(contents) > 0:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(UPLOAD_FOLDER, f"{username}_{filename}")
                    with open(file_path, 'wb') as f:
                        f.write(contents)
                    temp_files_to_cleanup.append(file_path)
                    file_size = len(contents) / 1024
                    logging.info(f"Successfully uploaded chat file: {filename} ({file_size:.2f} KB")
                    try:
                        pdf_content = get_uploaded_pdf_content(file_path, username, DummyProgress())
                        if pdf_content:
                            all_pdf_content.append(pdf_content)
                    except Exception as e:
                        logging.error(f"Error processing PDF content from {filename}: {str(e)}")
        if all_pdf_content:
            merged_content = "\n\n".join(all_pdf_content)
            user_data.file_data = user_data.encrypt_data(merged_content)
            return {'status': f'Successfully processed {len(all_pdf_content)} files'}
        else:
            return {'status': 'No valid PDF files processed'}
    finally:
        await safe_file_cleanup(temp_files_to_cleanup)

class HandleQueryData(BaseModel):
    query: str
    history: list = []

@app.post("/api/handle_query", response_class=StreamingResponse)
async def handle_query(
    data: HandleQueryData,
    session_data: dict = Depends(require_login),
    session_id: str = Depends(get_session_id)
):
    username = session_data['username']
    worker_pid = os.getpid()
    logging.info(f"Worker {worker_pid}: Handling query for user: {username}, Session ID: {session_id}")
    query = data.query
    history = data.history
    if not query:
        raise HTTPException(status_code=400, detail='Please enter a message.')
    user_data = get_user_data(username)
    moderation_result = check_moderation(query)
    if moderation_result:
        raise HTTPException(status_code=400, detail=f"Content flagged: {moderation_result}")

    async def generate():
        try:
            conversation_history = user_data.memory.load_memory_variables({})["history"]
            similarity_threshold = 0.6
            for i in range(0, len(conversation_history), 2):
                if i + 1 >= len(conversation_history):
                    continue
                human_message = conversation_history[i]
                ai_message = conversation_history[i + 1]
                if isinstance(human_message, HumanMessage) and isinstance(ai_message, AIMessage):
                    previous_input = human_message.content
                    similarity = check_text_similarity(query, previous_input)
                    if similarity >= similarity_threshold:
                        yield f"data: {json.dumps({'response': ai_message.content})}\n\n"
                        return
            prompt = (
                f"You are BeaconMedicalAi AMA, a medical and legal assistant.\n\n"
                f"The user {username} asked: {query}\n\n"
                f"Provide answer according to the AMA Guides.\n"
            )
            try:
                if user_data.file_data:
                    file_content = user_data.decrypt_data(user_data.file_data)
                    if file_content:
                        prompt += f"File content:\n{file_content}\n\n"
                if user_data.report_data:
                    report_content = user_data.decrypt_data(user_data.report_data)
                    if report_content:
                        prompt += f"Generated report:\n{report_content}\n\n"
                if user_data.rebuttal_data:
                    rebuttal_content = user_data.decrypt_data(user_data.rebuttal_data)
                    if rebuttal_content:
                        prompt += f"Generated rebuttal:\n{rebuttal_content}\n\n"
            except Exception as e:
                logging.error(f"Error decrypting user data: {str(e)}")
            try:
                if current_vectorstore:
                    retrieved_docs = current_vectorstore.similarity_search(query, k=2)
                    retrieved_docs = rerank_documents(query, retrieved_docs, top_k=2)
                    retrieved_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    if retrieved_content:
                        prompt += f"Relevant AMA content:\n{retrieved_content}\n\n"
            except Exception as e:
                logging.error(f"Error retrieving AMA content: {str(e)}")
            prompt += """
                - If no rebuttal data, inform user to generate one if relevant.
                - If no report data, inform user to generate one if relevant.
                - If unrelated to AMA, guide user back to task.
                - Be specific and precise, provide AMA references for medical queries.
            """
            response = ""
            chunk_count = 0
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_client, prompt, model_pro='gpt', max_tokens=1000, stream=True)
                try:
                    stream = future.result(timeout=30)
                    for chunk in stream:
                        try:
                            chunk_text = chunk.choices[0].delta.content or ""
                            response += chunk_text
                            chunk_count += 1
                            if chunk_count % 5 == 0:
                                yield f"data: {json.dumps({'response': response})}\n\n"
                        except:
                            continue
                    yield f"data: {json.dumps({'response': response})}\n\n"
                    user_data.memory.save_context({"input": query}, {"output": response})
                except concurrent.futures.TimeoutError:
                    yield f"data: {json.dumps({'response': 'Error: Request timed out. Please try again.'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'response': f'Error: {str(e)}'})}\n\n"
        except Exception as e:
            logging.error(f"Error in handle_query: {str(e)}")
            yield f"data: {json.dumps({'response': f'Error: {str(e)}'})}\n\n"

    return StreamingResponse(generate(), media_type='text/event-stream')

class DownloadChatHistoryData(BaseModel):
    history: list = []

@app.post("/api/download_chat_history")
async def download_chat_history_route(
    data: DownloadChatHistoryData,
    session_data: dict = Depends(require_login)
):
    username = session_data['username']
    history = data.history
    temp_file_path = save_chat_history(history, username)
    return FileResponse(temp_file_path, filename='chat_history.txt')

@app.get("/favicon.ico")
async def favicon():
    return FileResponse('favicon.ico', media_type='image/x-icon')

@app.post("/api/cancel_generation")
async def cancel_generation(session_data: dict = Depends(require_login)):
    GENERATION_CANCELLED = get_generation_cancelled()
    GENERATION_RUNNING = get_generation_running()
    GENERATION_CANCELLED.set()
    GENERATION_RUNNING.clear()
    return {'status': 'Generation cancelled'}

@app.get("/api/generation_status")
async def generation_status(session_data: dict = Depends(require_login)):
    GENERATION_RUNNING = get_generation_running()
    GENERATION_CANCELLED = get_generation_cancelled()
    return {
        'running': GENERATION_RUNNING.is_set(),
        'cancelled': GENERATION_CANCELLED.is_set()
    }

@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return JSONResponse({'error': 'Not found'}, status_code=404)

@app.exception_handler(500)
async def internal_error(request: Request, exc: HTTPException):
    logging.error(f"Internal server error: {str(exc)}")
    return JSONResponse({'error': 'Internal server error'}, status_code=500)

def cleanup_temp_files():
    """Clean up temporary files on application shutdown."""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
            logging.info(f"Cleaned up upload folder: {UPLOAD_FOLDER}")
        session_manager.cleanup_old_sessions()
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {str(e)}")

import atexit
atexit.register(cleanup_temp_files)

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 80))
    logging.info(f"Starting FastAPI app on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)