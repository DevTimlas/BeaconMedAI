import sys
import os
import asyncio
import threading
import tempfile
import markdown
import shutil
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, Response, stream_with_context, session
from flask_session import Session
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

app = Flask(__name__)

# FIXED SESSION CONFIGURATION FOR GUNICORN
# Create a shared directory for all workers to access
SHARED_SESSION_DIR = '/tmp/flask_sessions'  # Use absolute path for shared access
if not os.path.exists(SHARED_SESSION_DIR):
    os.makedirs(SHARED_SESSION_DIR, exist_ok=True)

app.secret_key = os.environ.get('SECRET_KEY', 'your_fixed_secret_key_here_change_in_production')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SHARED_SESSION_DIR  # Shared directory
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_FILE_THRESHOLD'] = 500
app.config['SESSION_USE_SIGNER'] = True  # Add signature validation
app.config['SESSION_KEY_PREFIX'] = 'flask_session:'  # Add prefix for identification
app.config['SESSION_FILE_MODE'] = 0o600  # Secure file permissions

# Initialize session AFTER all config is set
Session(app)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
ALLOWED_EXTENSIONS = {'.zip', '.pdf', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.jpg', '.jpeg', '.png'}

# Global variables - these need to be worker-safe
AUTH_CREDENTIALS = {
    "michael rice": "4321",
    "Tim": "1038",
    "Tester": "Test012"
}

# Use threading.local for worker-specific state
worker_state = threading.local()

def get_generation_running():
    """Get worker-specific generation running state."""
    if not hasattr(worker_state, 'generation_running'):
        worker_state.generation_running = threading.Event()
    return worker_state.generation_running

def get_generation_cancelled():
    """Get worker-specific generation cancelled state."""
    if not hasattr(worker_state, 'generation_cancelled'):
        worker_state.generation_cancelled = threading.Event()
    return worker_state.generation_cancelled

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def check_text_similarity(text1, text2, method='cosine'):
    """Check similarity between two texts using cosine similarity."""
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
    """Rerank retrieved documents based on embedding similarity."""
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
    """Mimic Gradio's progress for Flask compatibility."""
    def __call__(self, value=None, desc=None, total=None, unit=None, track_tqdm=False):
        if desc:
            logging.info(f"Progress: {desc} ({value}/{total} {unit})")
        return self

    def tqdm(self, iterable, desc=None, total=None):
        logging.info(f"Processing {desc} with {len(iterable) if total is None else total} items")
        return iterable

def safe_save_uploaded_file(file, file_path, temp_files_to_cleanup):
    """
    Save an uploaded FileStorage object to disk.
    Returns: (success: bool, error_message: str, file_size_kb: float)
    """
    try:
        # Log file stream state
        logging.info(f"Attempting to save file: {file.filename}, stream_closed: {file.stream.closed if hasattr(file, 'stream') else 'N/A'}, stream_type: {type(file.stream) if hasattr(file, 'stream') else 'N/A'}")
        
        # Handle closed streams by reading from the file object directly
        try:
            if hasattr(file, 'stream') and file.stream.closed:
                # Try to read the file content from the file object itself
                logging.warning(f"File stream is closed for {file.filename}, attempting alternative save method")
                
                # Reset stream position if possible
                if hasattr(file, 'seek'):
                    try:
                        file.seek(0)
                    except (OSError, ValueError):
                        pass
                
                # Try to read file content directly
                file_content = file.read()
                if not file_content:
                    logging.error(f"No content available for {file.filename}")
                    return False, "No file content available", 0
                
                # Write content to file
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                
                temp_files_to_cleanup.append(file_path)
                
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    file_size = os.path.getsize(file_path) / 1024
                    logging.info(f"File saved successfully via alternative method: {file_path}, size: {file_size:.2f} KB")
                    return True, None, file_size
                else:
                    logging.error(f"Alternative save method failed for {file.filename}")
                    return False, "Alternative save method failed", 0
            else:
                # Normal save method for open streams
                file.save(file_path)
                temp_files_to_cleanup.append(file_path)
                
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    file_size = os.path.getsize(file_path) / 1024
                    logging.info(f"File saved successfully: {file_path}, size: {file_size:.2f} KB")
                    return True, None, file_size
                else:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if file_path in temp_files_to_cleanup:
                        temp_files_to_cleanup.remove(file_path)
                    logging.error(f"Failed to save file {file.filename}: Empty or invalid file")
                    return False, "Empty or invalid file", 0
                    
        except Exception as save_error:
            logging.error(f"Error during file save for {file.filename}: {str(save_error)}")
            
            # Final fallback: try to copy from werkzeug FileStorage
            try:
                import shutil
                if hasattr(file, 'stream'):
                    with open(file_path, 'wb') as dest_file:
                        file.stream.seek(0)
                        shutil.copyfileobj(file.stream, dest_file)
                    
                    temp_files_to_cleanup.append(file_path)
                    
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        file_size = os.path.getsize(file_path) / 1024
                        logging.info(f"File saved successfully via fallback method: {file_path}, size: {file_size:.2f} KB")
                        return True, None, file_size
                    
            except Exception as fallback_error:
                logging.error(f"Fallback save method also failed for {file.filename}: {str(fallback_error)}")
            
            return False, f"All save methods failed: {str(save_error)}", 0
                
    except Exception as e:
        logging.error(f"Error saving file {file.filename}: {str(e)}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return False, f"Error saving file: {str(e)}", 0

def safe_file_cleanup(file_paths):
    """Safely clean up files with error handling."""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {str(e)}")

async def collect_file_results(file_paths, username, progress, cancelled):
    """Collect results from process_file with improved error handling."""
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

        if not callable(process_file):
            logging.error("process_file is not callable")
            return ["Error: process_file is not a function"]

        try:
            result = process_file(file_paths, username, progress or DummyProgress(), cancelled)
            
            if result is None:
                logging.error("process_file returned None")
                return ["Error: process_file returned None"]

            if hasattr(result, '__aiter__'):
                async for item in result:
                    if cancelled.is_set():
                        logging.info("File processing cancelled by user")
                        return ["Generation cancelled."]
                    
                    if item is None:
                        logging.warning("process_file yielded None item, skipping")
                        continue
                        
                    if isinstance(item, str) and (item == "Generation cancelled." or item.startswith("Error:")):
                        logging.error(f"Process file returned error: {item}")
                        return [item]
                    
                    results.append(str(item))
                    logging.info(f"Collected result: {str(item)[:100]}...")
            else:
                if isinstance(result, str):
                    if result == "Generation cancelled." or result.startswith("Error:"):
                        return [result]
                    results.append(result)
                else:
                    logging.error(f"process_file returned unexpected type: {type(result)}")
                    return [f"Error: Unexpected return type from process_file: {type(result)}"]

        except Exception as e:
            logging.error(f"Error calling process_file: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return [f"Error in file processing: {str(e)}"]

        if not results:
            logging.error("process_file yielded no valid results")
            return ["Error: No valid results from file processing"]
        
        logging.info(f"Successfully collected {len(results)} results from file processing")
        return results
        
    except Exception as e:
        logging.error(f"Error in collect_file_results: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return [f"Error: {str(e)}"]

async def collect_combine_results(reports, username, progress, cancelled):
    """Collect results from get_combine_reports with improved error handling."""
    results = []
    try:
        if not reports:
            logging.error("No reports provided to get_combine_reports")
            return ["Error: No reports provided"]
        
        logging.info(f"Combining {len(reports)} reports")
        
        result = get_combine_reports(reports, username, progress or DummyProgress(), cancelled)
        
        if result is None:
            logging.error("get_combine_reports returned None")
            return ["Error: get_combine_reports returned None"]
        
        if hasattr(result, '__aiter__'):
            async for item in result:
                if cancelled.is_set():
                    logging.info("Report combination cancelled by user")
                    return ["Generation cancelled."]
                
                if item is None:
                    logging.warning("get_combine_reports yielded None item, skipping")
                    continue
                    
                if isinstance(item, str) and (item == "Generation cancelled." or item.startswith("Error:")):
                    logging.error(f"Combine reports returned error: {item}")
                    return [item]
                
                results.append(str(item))
                logging.info(f"Collected combined result: {str(item)[:100]}...")
        else:
            if isinstance(result, str):
                if result == "Generation cancelled." or result.startswith("Error:"):
                    return [result]
                results.append(result)
            else:
                logging.error(f"get_combine_reports returned unexpected type: {type(result)}")
                return [f"Error: Unexpected return type from get_combine_reports: {type(result)}"]
        
        if not results:
            logging.error("get_combine_reports yielded no valid results")
            return ["Error: No valid results from report combination"]
        
        logging.info(f"Successfully collected {len(results)} combined results")
        return results
        
    except Exception as e:
        logging.error(f"Error in collect_combine_results: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return [f"Error: {str(e)}"]

# Session validation middleware
@app.before_request
def validate_session():
    """Validate session before each request."""
    if request.endpoint in ['login', 'serve_index', 'favicon']:
        return  # Skip validation for these endpoints
    
    try:
        # Log session info for debugging
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        username = session.get('username')
        
        logging.debug(f"Worker {worker_pid}: Session validation - ID: {session_id}, User: {username}")
        
        # Force session to be marked as accessed/modified to ensure it's saved
        if username:
            session.permanent = True
            session.modified = True
            
    except Exception as e:
        logging.error(f"Error in session validation: {str(e)}")

@app.route('/')
def serve_index():
    try:
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        username = session.get('username', 'None')
        
        logging.info(f"Worker {worker_pid}: Index accessed - Session ID: {session_id}, User: {username}")
        return send_file('index.html')
    except Exception as e:
        logging.error(f"Error serving index: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if username in AUTH_CREDENTIALS and AUTH_CREDENTIALS[username] == password and validate_login(username, password):
            session.permanent = True
            session['username'] = username
            session.modified = True  # Force session to be saved
            
            worker_pid = os.getpid()
            session_id = session.get('_id', 'N/A')
            
            logging.info(f"Worker {worker_pid}: User {username} logged in - Session ID: {session_id}")
            return jsonify({'status': 'success', 'username': username})
        
        logging.error(f"Failed login attempt for username: {username}")
        return jsonify({'status': 'error', 'message': 'Invalid username or password'}), 401
    except Exception as e:
        logging.error(f"Error in login: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        username = session.get('username')
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        
        session.clear()  # Clear entire session
        session.modified = True
        
        logging.info(f"Worker {worker_pid}: User {username} logged out - Session ID: {session_id}")
        return jsonify({'status': 'success', 'message': 'Logged out'})
    except Exception as e:
        logging.error(f"Error in logout: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Logout failed'}), 500

@app.route('/api/get_username', methods=['GET'])
def get_username():
    try:
        username = session.get('username')
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        
        logging.info(f"Worker {worker_pid}: Checking session - User: {username}, Session ID: {session_id}")
        
        if username:
            # Refresh session
            session.permanent = True
            session.modified = True
            return jsonify({'username': username})
        
        return jsonify({'username': None})
    except Exception as e:
        logging.error(f"Error getting username: {str(e)}")
        return jsonify({'username': None})

@app.route('/api/update_vectorstore', methods=['POST'])
def update_vectorstore():
    if 'username' not in session:
        logging.error("Vectorstore update attempted without login")
        return jsonify({'status': 'error', 'message': 'Please log in first.'}), 401
    
    try:
        data = request.json
        edition = data.get('edition', '4th Edition')
        initialize_vectorstore_for_edition(edition)
        logging.info(f"Vectorstore updated to {edition}")
        return jsonify({'status': f"Using AMA Guides {edition}"})
    except Exception as e:
        logging.error(f"Error updating vectorstore: {e}")
        return jsonify({'status': f"Error: Failed to load {edition} vectorstore"}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    if 'username' not in session:
        logging.error("Report generation attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    
    username = session['username']
    worker_pid = os.getpid()
    session_id = session.get('_id', 'N/A')
    
    logging.info(f"Worker {worker_pid}: Generating report for user: {username}, Session ID: {session_id}")
    
    GENERATION_RUNNING = get_generation_running()
    GENERATION_CANCELLED = get_generation_cancelled()
    
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()

    try:
        files = request.files.getlist('files')
        text_input = request.form.get('text', '')
        edition = request.form.get('edition', '4th Edition')

        # Enhanced file validation and logging
        valid_files = []
        for file in files:
            if file and file.filename:
                logging.info(f"Received file: {file.filename}")
                logging.info(f"File content type: {file.content_type}")
                logging.info(f"File content length: {file.content_length}")
                
                # Check if file has content
                if hasattr(file, 'content_length') and file.content_length == 0:
                    logging.warning(f"File {file.filename} has zero content length")
                    continue
                
                # Try to peek at file size
                try:
                    if hasattr(file, 'stream'):
                        current_pos = file.stream.tell() if hasattr(file.stream, 'tell') else 0
                        file.stream.seek(0, 2)  # Seek to end
                        file_size = file.stream.tell()
                        file.stream.seek(current_pos)  # Seek back
                        logging.info(f"File {file.filename} actual size: {file_size} bytes")
                        
                        if file_size == 0:
                            logging.warning(f"File {file.filename} is empty")
                            continue
                except Exception as e:
                    logging.warning(f"Could not determine size for {file.filename}: {str(e)}")
                
                valid_files.append(file)
            elif file:
                logging.warning(f"File object received but no filename: {file}")

        initialize_vectorstore_for_edition(edition)
        user_data = get_user_data(username)

        if not valid_files and not text_input.strip():
            GENERATION_RUNNING.clear()
            logging.error("No valid input provided for report generation")
            return jsonify({'error': 'Please upload at least one valid document or provide text input.'}), 400

        def generate():
            all_reports = []
            file_paths = []
            temp_files_to_cleanup = []
            
            try:
                # Handle file uploads
                if valid_files:
                    for file in valid_files:
                        if allowed_file(file.filename):
                            filename = secure_filename(file.filename)
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{username}_{filename}")
                            
                            success, error_msg, file_size = safe_save_uploaded_file(file, file_path, temp_files_to_cleanup)
                            
                            if success:
                                file_paths.append(file_path)
                                logging.info(f"Successfully uploaded: {filename} ({file_size:.2f} KB)")
                            else:
                                logging.error(f"Failed to upload {filename}: {error_msg}")
                                yield f"data: {json.dumps({'report': f'Error uploading {filename}: {error_msg}'})}\n\n"
                                return
                        else:
                            error_msg = f"Invalid file type: {file.filename}"
                            logging.error(error_msg)
                            yield f"data: {json.dumps({'report': f'Error: {error_msg}'})}\n\n"
                            return

                    # Process uploaded files
                    if file_paths:
                        logging.info(f"Processing {len(file_paths)} files")
                        partial_results = asyncio.run(collect_file_results(file_paths, username, DummyProgress(), GENERATION_CANCELLED))
                        
                        for partial_result in partial_results:
                            if GENERATION_CANCELLED.is_set():
                                GENERATION_RUNNING.clear()
                                yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                                return
                                
                            if partial_result == "Generation cancelled." or partial_result.startswith("Error:"):
                                GENERATION_RUNNING.clear()
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
                if text_input.strip():
                    try:
                        moderation_result = check_moderation(text_input)
                        if moderation_result:
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'report': f'Content flagged: {moderation_result}'})}\n\n"
                            return
                        
                        logging.info("Processing text input for AMA report")
                        text_report = asyncio.run(get_ama_impairment_report(text_input, username, DummyProgress(), GENERATION_CANCELLED, current_vectorstore))
                        
                        if GENERATION_CANCELLED.is_set():
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                            return
                        
                        if text_report and not text_report.startswith("Error:"):
                            if not valid_files:
                                logging.info("Combining text report")
                                partial_results = asyncio.run(collect_combine_results([text_report], username, DummyProgress(), GENERATION_CANCELLED))
                                
                                for combined_report in partial_results:
                                    if GENERATION_CANCELLED.is_set():
                                        GENERATION_RUNNING.clear()
                                        yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                                        return
                                        
                                    if combined_report == "Generation cancelled." or combined_report.startswith("Error:"):
                                        GENERATION_RUNNING.clear()
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
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'report': error_msg})}\n\n"
                            return
                            
                    except Exception as e:
                        logging.error(f"Error processing text input: {str(e)}")
                        GENERATION_RUNNING.clear()
                        yield f"data: {json.dumps({'report': f'Error processing text: {str(e)}'})}\n\n"
                        return

                # Final processing
                if not all_reports:
                    GENERATION_RUNNING.clear()
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
                        else:
                            logging.warning("PDF buffer is None")
                    except Exception as pdf_error:
                        logging.error(f"Error creating PDF: {str(pdf_error)}")

                    GENERATION_RUNNING.clear()
                    final_response = {'report': markdown.markdown(final_report_text)}
                    if pdf_base64:
                        final_response['pdf'] = pdf_base64
                    
                    logging.info(f"Streaming final report: {final_report_text[:50]}...")
                    yield f"data: {json.dumps(final_response)}\n\n"

                except Exception as e:
                    logging.error(f"Error in final processing: {str(e)}")
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'report': f'Error in final processing: {str(e)}'})}\n\n"

            except Exception as e:
                logging.error(f"Error in report generation: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                GENERATION_RUNNING.clear()
                yield f"data: {json.dumps({'report': f'Error: {str(e)}'})}\n\n"
            finally:
                safe_file_cleanup(temp_files_to_cleanup)

        return Response(stream_with_context(generate()), content_type='text/event-stream')
        
    except Exception as e:
        logging.error(f"Error setting up report generation: {str(e)}")
        GENERATION_RUNNING.clear()
        return jsonify({'error': f'Setup error: {str(e)}'}), 500

@app.route('/api/generate_rebuttal', methods=['POST'])
def generate_rebuttal():
    if 'username' not in session:
        logging.error("Rebuttal generation attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    
    username = session['username']
    worker_pid = os.getpid()
    session_id = session.get('_id', 'N/A')
    
    logging.info(f"Worker {worker_pid}: Generating rebuttal for user: {username}, Session ID: {session_id}")
    
    GENERATION_RUNNING = get_generation_running()
    GENERATION_CANCELLED = get_generation_cancelled()
    
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()

    try:
        files = request.files.getlist('files')
        text_input = request.form.get('text', '')
        edition = request.form.get('edition', '4th Edition')

        # Log file stream state for debugging
        for file in files:
            if file and file.filename:
                logging.info(f"File: {file.filename}, stream_closed: {file.stream.closed if hasattr(file, 'stream') else 'N/A'}, stream_type: {type(file.stream) if hasattr(file, 'stream') else 'N/A'}")

        initialize_vectorstore_for_edition(edition)
        user_data = get_user_data(username)

        if not files and not text_input.strip():
            GENERATION_RUNNING.clear()
            logging.error("No input provided for rebuttal generation")
            return jsonify({'error': 'Please upload a document or provide text.'}), 400

        def generate():
            all_responses = []
            temp_files_to_cleanup = []
            
            try:
                if files:
                    file_paths = []
                    for file in files:
                        if file and file.filename and allowed_file(file.filename):
                            filename = secure_filename(file.filename)
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            
                            success, error_msg, file_size = safe_save_uploaded_file(file, file_path, temp_files_to_cleanup)
                            
                            if success:
                                file_paths.append(file_path)
                                logging.info(f"Successfully uploaded rebuttal file: {filename} ({file_size:.2f} KB)")
                            else:
                                logging.error(f"Failed to upload rebuttal file {filename}: {error_msg}")
                                yield f"data: {json.dumps({'rebuttal': f'Error uploading {filename}: {error_msg}'})}\n\n"
                                return
                        else:
                            error_msg = f"Invalid file: {file.filename if file and file.filename else 'Unknown file'}"
                            logging.error(error_msg)
                            yield f"data: {json.dumps({'rebuttal': f'Error: {error_msg}'})}\n\n"
                            return

                    if file_paths:
                        logging.info(f"Processing {len(file_paths)} files for rebuttal")
                        partial_results = asyncio.run(collect_file_results(file_paths, username, DummyProgress(), GENERATION_CANCELLED))
                        
                        for rebuttal_text in partial_results:
                            if GENERATION_CANCELLED.is_set():
                                GENERATION_RUNNING.clear()
                                yield f"data: {json.dumps({'rebuttal': 'Generation cancelled.'})}\n\n"
                                return
                                
                            if rebuttal_text == "Generation cancelled." or rebuttal_text.startswith("Error:"):
                                GENERATION_RUNNING.clear()
                                yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"
                                return
                            
                            all_responses.append(rebuttal_text)
                            try:
                                markdown_rebuttal = markdown.markdown(rebuttal_text)
                                logging.info(f"Streaming rebuttal: {rebuttal_text[:50]}...")
                                yield f"data: {json.dumps({'rebuttal': markdown_rebuttal})}\n\n"
                            except Exception as e:
                                logging.error(f"Error converting rebuttal to markdown: {str(e)}")
                                yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"

                if text_input.strip():
                    try:
                        moderation_result = check_moderation(text_input)
                        if moderation_result:
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'rebuttal': f'Content flagged: {moderation_result}'})}\n\n"
                            return
                        
                        logging.info("Processing text input for rebuttal")
                        rebuttal_text = asyncio.run(process_batch_for_rebuttal([text_input], username, DummyProgress(), GENERATION_CANCELLED, current_vectorstore))
                        
                        if GENERATION_CANCELLED.is_set():
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'rebuttal': 'Generation cancelled.'})}\n\n"
                            return
                        
                        if rebuttal_text and rebuttal_text != "Generation cancelled." and not rebuttal_text.startswith("Error:"):
                            all_responses.append(rebuttal_text)
                            try:
                                markdown_rebuttal = markdown.markdown(rebuttal_text)
                                logging.info(f"Streaming text rebuttal: {rebuttal_text[:50]}...")
                                yield f"data: {json.dumps({'rebuttal': markdown_rebuttal})}\n\n"
                            except Exception as e:
                                logging.error(f"Error converting text rebuttal to markdown: {str(e)}")
                                yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"
                        else:
                            error_msg = rebuttal_text if rebuttal_text else "Failed to generate text rebuttal"
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'rebuttal': error_msg})}\n\n"
                            return
                            
                    except Exception as e:
                        logging.error(f"Error processing text input for rebuttal: {str(e)}")
                        GENERATION_RUNNING.clear()
                        yield f"data: {json.dumps({'rebuttal': f'Error processing text: {str(e)}'})}\n\n"
                        return

                if not all_responses:
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'rebuttal': 'No rebuttal generated.'})}\n\n"
                    return

                final_rebuttal_text = user_data.decrypt_data(user_data.rebuttal_data) if user_data.rebuttal_data else all_responses[-1]
                
                if final_rebuttal_text and final_rebuttal_text != "Generation cancelled.":
                    try:
                        if not user_data.rebuttal_data and all_responses:
                            user_data.rebuttal_data = user_data.encrypt_data(all_responses[-1])
                            logging.info("Rebuttal data encrypted and saved")
                        
                        pdf_base64 = None
                        try:
                            pdf_buffer = create_pdf_from_markdown(final_rebuttal_text)
                            if pdf_buffer:
                                pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                                logging.info("Rebuttal PDF generated successfully")
                            else:
                                logging.warning("Rebuttal PDF buffer is None")
                        except Exception as pdf_error:
                            logging.error(f"Error creating rebuttal PDF: {str(pdf_error)}")

                        GENERATION_RUNNING.clear()
                        final_response = {'rebuttal': markdown.markdown(final_rebuttal_text)}
                        if pdf_base64:
                            final_response['pdf'] = pdf_base64
                        
                        logging.info(f"Streaming final rebuttal: {final_rebuttal_text[:50]}...")
                        yield f"data: {json.dumps(final_response)}\n\n"
                        
                    except Exception as e:
                        logging.error(f"Error in final rebuttal processing: {str(e)}")
                        GENERATION_RUNNING.clear()
                        yield f"data: {json.dumps({'rebuttal': f'Error in final processing: {str(e)}'})}\n\n"
                else:
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'rebuttal': final_rebuttal_text or 'No rebuttal text available'})}\n\n"

            except Exception as e:
                logging.error(f"Error in rebuttal generation: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                GENERATION_RUNNING.clear()
                yield f"data: {json.dumps({'rebuttal': f'Error: {str(e)}'})}\n\n"
            finally:
                safe_file_cleanup(temp_files_to_cleanup)

        return Response(stream_with_context(generate()), content_type='text/event-stream')
        
    except Exception as e:
        logging.error(f"Error setting up rebuttal generation: {str(e)}")
        GENERATION_RUNNING.clear()
        return jsonify({'error': f'Setup error: {str(e)}'}), 500

@app.route('/api/reset_reports', methods=['POST'])
def reset_reports():
    if 'username' not in session:
        logging.error("Reset reports attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    
    try:
        username = session['username']
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        
        user_data = get_user_data(username)
        user_data.file_data = b""
        user_data.report_data = b""
        user_data.rebuttal_data = b""
        
        logging.info(f"Worker {worker_pid}: Reports reset for user: {username}, Session ID: {session_id}")
        return jsonify({'status': 'Reports reset'})
    except Exception as e:
        logging.error(f"Error resetting reports: {str(e)}")
        return jsonify({'error': f'Error resetting reports: {str(e)}'}), 500

@app.route('/api/process_chat_files', methods=['POST'])
def process_chat_files():
    if 'username' not in session:
        logging.error("Chat file processing attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    
    username = session['username']
    worker_pid = os.getpid()
    session_id = session.get('_id', 'N/A')
    
    logging.info(f"Worker {worker_pid}: Processing chat files for user: {username}, Session ID: {session_id}")
    temp_files_to_cleanup = []
    
    try:
        files = request.files.getlist('files')
        user_data = get_user_data(username)
        all_pdf_content = []

        # Log file stream state for debugging
        for file in files:
            if file and file.filename:
                logging.info(f"File: {file.filename}, stream_closed: {file.stream.closed if hasattr(file, 'stream') else 'N/A'}, stream_type: {type(file.stream) if hasattr(file, 'stream') else 'N/A'}")

        for file in files:
            if file and file.filename and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                success, error_msg, file_size = safe_save_uploaded_file(file, file_path, temp_files_to_cleanup)
                
                if success:
                    logging.info(f"Successfully uploaded chat file: {filename} ({file_size:.2f} KB)")
                    
                    try:
                        pdf_content = get_uploaded_pdf_content(file_path, username, DummyProgress())
                        if pdf_content is None:
                            logging.error(f"Failed to process PDF: {file_path}")
                            continue
                        
                        all_pdf_content.append(pdf_content)
                        logging.info(f"Successfully processed PDF: {file_path}")
                    except Exception as e:
                        logging.error(f"Error processing PDF content from {filename}: {str(e)}")
                        continue
                else:
                    logging.error(f"Failed to upload chat file {filename}: {error_msg}")
                    continue

        if all_pdf_content:
            merged_content = "\n\n".join(all_pdf_content)
            user_data.file_data = user_data.encrypt_data(merged_content)
            logging.info(f"Stored {len(all_pdf_content)} PDF contents for user: {username}")
            return jsonify({'status': f'Successfully processed {len(all_pdf_content)} files'})
        else:
            return jsonify({'status': 'No valid PDF files processed'}), 400
            
    except Exception as e:
        logging.error(f"Error in process_chat_files: {str(e)}")
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500
    finally:
        safe_file_cleanup(temp_files_to_cleanup)

@app.route('/api/handle_query', methods=['POST'])
def handle_query():
    if 'username' not in session:
        logging.error("Query handling attempted without login")
        return jsonify({'response': 'Please log in first.'}), 401
    
    username = session['username']
    worker_pid = os.getpid()
    session_id = session.get('_id', 'N/A')
    
    logging.info(f"Worker {worker_pid}: Handling query for user: {username}, Session ID: {session_id}")
    
    try:
        data = request.json
        query = data.get('query')
        history = data.get('history', [])

        if not query:
            logging.error("Empty query received")
            return jsonify({'response': 'Please enter a message.'}), 400

        user_data = get_user_data(username)
        moderation_result = check_moderation(query)
        if moderation_result:
            logging.error(f"Query flagged: {moderation_result}")
            return jsonify({'response': f"Content flagged: {moderation_result}"}), 400

        def generate():
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
                        similarity = check_text_similarity(query, previous_input, method='cosine')
                        logging.info(f"Similarity with previous input {i // 2 + 1}: {similarity}")
                        
                        if similarity >= similarity_threshold:
                            logging.info(f"Returning stored response for similar input (similarity: {similarity})")
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

                if history:
                    try:
                        history_text = "\n\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
                        prompt = f"{history_text}\n\n{prompt}"
                    except Exception as e:
                        logging.error(f"Error processing conversation history: {str(e)}")

                response = ""
                chunk_count = 0

                def stream_client():
                    try:
                        return init_client(
                            prompt,
                            model_pro='gpt',
                            max_tokens=1000,
                            stream=True
                        )
                    except Exception as e:
                        logging.error(f"Error initializing client: {str(e)}")
                        raise

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(stream_client)
                    try:
                        stream = future.result(timeout=30)
                        
                        for chunk in stream:
                            try:
                                chunk_text = chunk.choices[0].delta.content or ""
                                response += chunk_text
                                chunk_count += 1
                                
                                if chunk_count % 5 == 0:
                                    yield f"data: {json.dumps({'response': response})}\n\n"
                            except Exception as e:
                                logging.error(f"Error processing chunk: {str(e)}")
                                continue
                        
                        yield f"data: {json.dumps({'response': response})}\n\n"

                        try:
                            user_data.memory.save_context({"input": query}, {"output": response})
                            logging.info("Conversation saved to memory")
                        except Exception as e:
                            logging.error(f"Error saving to memory: {str(e)}")

                    except concurrent.futures.TimeoutError:
                        logging.error("Timeout in stream_client call")
                        yield f"data: {json.dumps({'response': 'Error: Request to language model timed out. Please try again.'})}\n\n"
                    except Exception as e:
                        logging.error(f"Error in streaming client: {str(e)}")
                        yield f"data: {json.dumps({'response': f'Error: {str(e)}'})}\n\n"

            except Exception as e:
                logging.error(f"Error in handle_query: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                yield f"data: {json.dumps({'response': f'Error: {str(e)}'})}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')
        
    except Exception as e:
        logging.error(f"Error setting up query handling: {str(e)}")
        return jsonify({'error': f'Setup error: {str(e)}'}), 500

@app.route('/api/download_chat_history', methods=['POST'])
def download_chat_history_route():
    if 'username' not in session:
        logging.error("Chat history download attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    
    try:
        username = session['username']
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        
        data = request.json
        history = data.get('history', [])
        temp_file_path = save_chat_history(history, username)
        
        logging.info(f"Worker {worker_pid}: Chat history downloaded for user: {username}, Session ID: {session_id}")
        return send_file(temp_file_path, as_attachment=True, download_name='chat_history.txt')
    except Exception as e:
        logging.error(f"Error downloading chat history: {str(e)}")
        return jsonify({'error': f'Error downloading chat history: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    try:
        return send_file('favicon.ico', mimetype='image/x-icon')
    except Exception:
        return '', 204

@app.route('/api/cancel_generation', methods=['POST'])
def cancel_generation():
    if 'username' not in session:
        return jsonify({'error': 'Please log in first.'}), 401
    
    try:
        GENERATION_CANCELLED = get_generation_cancelled()
        GENERATION_RUNNING = get_generation_running()
        
        GENERATION_CANCELLED.set()
        GENERATION_RUNNING.clear()
        
        worker_pid = os.getpid()
        session_id = session.get('_id', 'N/A')
        
        logging.info(f"Worker {worker_pid}: Generation cancelled by user: {session['username']}, Session ID: {session_id}")
        return jsonify({'status': 'Generation cancelled'})
    except Exception as e:
        logging.error(f"Error cancelling generation: {str(e)}")
        return jsonify({'error': f'Error cancelling generation: {str(e)}'}), 500

@app.route('/api/generation_status', methods=['GET'])
def generation_status():
    if 'username' not in session:
        return jsonify({'error': 'Please log in first.'}), 401
    
    try:
        GENERATION_RUNNING = get_generation_running()
        GENERATION_CANCELLED = get_generation_cancelled()
        
        return jsonify({
            'running': GENERATION_RUNNING.is_set(),
            'cancelled': GENERATION_CANCELLED.is_set()
        })
    except Exception as e:
        logging.error(f"Error getting generation status: {str(e)}")
        return jsonify({'error': f'Error getting status: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    logging.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

def cleanup_temp_files():
    """Clean up temporary files and session files on application shutdown."""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
            logging.info(f"Cleaned up upload folder: {UPLOAD_FOLDER}")
        if os.path.exists(SHARED_SESSION_DIR):
            # Only clean up session files older than 25 hours
            import time
            current_time = time.time()
            for filename in os.listdir(SHARED_SESSION_DIR):
                file_path = os.path.join(SHARED_SESSION_DIR, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 25 * 3600:  # 25 hours in seconds
                        try:
                            os.remove(file_path)
                            logging.info(f"Cleaned up old session file: {file_path}")
                        except Exception as e:
                            logging.error(f"Error cleaning session file {file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {str(e)}")

import atexit
atexit.register(cleanup_temp_files)

if __name__ == '__main__':
    try:
        initialize_vectorstore_for_edition("4th Edition")
        port = int(os.environ.get("PORT", 80))
        logging.info(f"Starting Flask app on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logging.error(f"Error starting application: {str(e)}")
        sys.exit(1)