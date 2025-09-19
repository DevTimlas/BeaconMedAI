import sys
import os
import asyncio
import threading
import tempfile
import markdown
import shutil
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response, stream_with_context, session
from werkzeug.utils import secure_filename
from auth.authentication import validate_login, ACTIVE_SESSIONS, SESSION_TIMEOUT
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
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'.zip', '.pdf', '.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.jpg', '.jpeg', '.png'}

# Global variables
AUTH_CREDENTIALS = {
    "michael rice": "4321",
    "Tim": "1038",
    "Tester": "Test012"
}
GENERATION_RUNNING = threading.Event()
GENERATION_CANCELLED = threading.Event()

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

        # Validate all files exist and are readable
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logging.error(f"File does not exist: {file_path}")
                return [f"Error: File {file_path} does not exist"]
            
            try:
                file_size = os.path.getsize(file_path) / 1024  # Size in KB
                logging.info(f"Processing file: {file_path}, size: {file_size:.2f} KB")
            except Exception as e:
                logging.error(f"Error accessing file {file_path}: {str(e)}")
                return [f"Error: Cannot access file {file_path}"]

        if not callable(process_file):
            logging.error("process_file is not callable")
            return ["Error: process_file is not a function"]

        # Call process_file with proper error handling
        try:
            result = process_file(file_paths, username, progress or DummyProgress(), cancelled)
            
            if result is None:
                logging.error("process_file returned None")
                return ["Error: process_file returned None"]

            # Handle async generator
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
                # Handle non-async generator case
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
        
        # Handle async generator
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
            # Handle non-async case
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

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if username in AUTH_CREDENTIALS and AUTH_CREDENTIALS[username] == password and validate_login(username, password):
            session['username'] = username
            ACTIVE_SESSIONS[username] = datetime.now()
            logging.info(f"User {username} logged in")
            return jsonify({'status': 'success', 'username': username})
        
        logging.error(f"Failed login attempt for username: {username}")
        return jsonify({'status': 'error', 'message': 'Invalid username or password'}), 401
    except Exception as e:
        logging.error(f"Error in login: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    try:
        username = session.pop('username', None)
        if username in ACTIVE_SESSIONS:
            del ACTIVE_SESSIONS[username]
            logging.info(f"User {username} logged out")
        return jsonify({'status': 'success', 'message': 'Logged out'})
    except Exception as e:
        logging.error(f"Error in logout: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Logout failed'}), 500

@app.route('/api/get_username', methods=['GET'])
def get_username():
    try:
        username = session.get('username')
        if username and username in ACTIVE_SESSIONS:
            if (datetime.now() - ACTIVE_SESSIONS[username]).total_seconds() < SESSION_TIMEOUT:
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
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()

    try:
        files = request.files.getlist('files')
        text_input = request.form.get('text', '')
        edition = request.form.get('edition', '4th Edition')

        initialize_vectorstore_for_edition(edition)
        user_data = get_user_data(username)

        if not files and not text_input.strip():
            GENERATION_RUNNING.clear()
            logging.error("No input provided for report generation")
            return jsonify({'error': 'Please upload at least one document or provide text input.'}), 400

        def generate():
            all_reports = []
            file_paths = []
            temp_files_to_cleanup = []
            
            try:
                # Handle file uploads
                if files:
                    for file in files:
                        if file and file.filename and allowed_file(file.filename):
                            filename = secure_filename(file.filename)
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            
                            try:
                                file.save(file_path)
                                temp_files_to_cleanup.append(file_path)
                                
                                if os.path.exists(file_path):
                                    file_size = os.path.getsize(file_path) / 1024
                                    logging.info(f"Saved file: {file_path}, size: {file_size:.2f} KB")
                                    file_paths.append(file_path)
                                else:
                                    logging.error(f"File was not saved properly: {file_path}")
                                    yield f"data: {json.dumps({'report': f'Error: Failed to save file {filename}'})}\n\n"
                                    return
                            except Exception as e:
                                logging.error(f"Error saving file {filename}: {str(e)}")
                                yield f"data: {json.dumps({'report': f'Error saving file {filename}: {str(e)}'})}\n\n"
                                return
                        else:
                            error_msg = f"Invalid file: {file.filename if file and file.filename else 'Unknown file'}"
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
                            # If no files were uploaded, combine the text report
                            if not files:
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

                # Get the final report
                final_report_text = all_reports[-1]
                
                try:
                    # Save report data
                    user_data.report_data = user_data.encrypt_data(final_report_text)
                    logging.info("Report data encrypted and saved")
                    
                    # Create PDF with error handling
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
                        # Continue without PDF

                    # Final response
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
                # Always cleanup files
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
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()

    try:
        files = request.files.getlist('files')
        text_input = request.form.get('text', '')
        edition = request.form.get('edition', '4th Edition')

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
                # Handle file uploads
                if files:
                    file_paths = []
                    for file in files:
                        if file and file.filename and allowed_file(file.filename):
                            filename = secure_filename(file.filename)
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            
                            try:
                                file.save(file_path)
                                temp_files_to_cleanup.append(file_path)
                                
                                if os.path.exists(file_path):
                                    file_size = os.path.getsize(file_path) / 1024
                                    logging.info(f"Saved rebuttal file: {file_path}, size: {file_size:.2f} KB")
                                    file_paths.append(file_path)
                                else:
                                    logging.error(f"Rebuttal file was not saved properly: {file_path}")
                                    yield f"data: {json.dumps({'rebuttal': f'Error: Failed to save file {filename}'})}\n\n"
                                    return
                            except Exception as e:
                                logging.error(f"Error saving rebuttal file {filename}: {str(e)}")
                                yield f"data: {json.dumps({'rebuttal': f'Error saving file {filename}: {str(e)}'})}\n\n"
                                return
                        else:
                            error_msg = f"Invalid file: {file.filename if file and file.filename else 'Unknown file'}"
                            logging.error(error_msg)
                            yield f"data: {json.dumps({'rebuttal': f'Error: {error_msg}'})}\n\n"
                            return

                    # Process uploaded files for rebuttal
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

                # Handle text input for rebuttal
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

                # Final rebuttal processing
                if not all_responses:
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'rebuttal': 'No rebuttal generated.'})}\n\n"
                    return

                # Get final rebuttal text
                final_rebuttal_text = user_data.decrypt_data(user_data.rebuttal_data) if user_data.rebuttal_data else all_responses[-1]
                
                if final_rebuttal_text and final_rebuttal_text != "Generation cancelled.":
                    try:
                        # Save rebuttal data if from new generation
                        if not user_data.rebuttal_data and all_responses:
                            user_data.rebuttal_data = user_data.encrypt_data(all_responses[-1])
                            logging.info("Rebuttal data encrypted and saved")
                        
                        # Create PDF with error handling
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
                            # Continue without PDF

                        # Final response
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
                # Always cleanup files
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
        user_data = get_user_data(username)
        user_data.file_data = b""
        user_data.report_data = b""
        user_data.rebuttal_data = b""
        logging.info(f"Reports reset for user: {username}")
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
    temp_files_to_cleanup = []
    
    try:
        files = request.files.getlist('files')
        user_data = get_user_data(username)
        all_pdf_content = []

        for file in files:
            if file and file.filename and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(file_path)
                    temp_files_to_cleanup.append(file_path)
                    
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024
                        logging.info(f"Processing chat file: {file_path}, size: {file_size:.2f} KB")
                        
                        pdf_content = get_uploaded_pdf_content(file_path, username, DummyProgress())
                        if pdf_content is None:
                            logging.error(f"Failed to process PDF: {file_path}")
                            continue
                        
                        all_pdf_content.append(pdf_content)
                        logging.info(f"Successfully processed PDF: {file_path}")
                    else:
                        logging.error(f"Chat file was not saved properly: {file_path}")
                        
                except Exception as e:
                    logging.error(f"Error processing chat file {filename}: {str(e)}")
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
        # Always cleanup files
        safe_file_cleanup(temp_files_to_cleanup)

@app.route('/api/handle_query', methods=['POST'])
def handle_query():
    if 'username' not in session:
        logging.error("Query handling attempted without login")
        return jsonify({'response': 'Please log in first.'}), 401
    
    username = session['username']
    
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
                # Check for similar previous queries
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

                # Construct prompt with context
                prompt = (
                    f"You are BeaconMedicalAi AMA, a medical and legal assistant.\n\n"
                    f"The user {username} asked: {query}\n\n"
                    f"Provide answer according to the AMA Guides.\n"
                )

                # Add user data context
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

                # Retrieve relevant AMA content
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

                # Add conversation history
                if history:
                    try:
                        history_text = "\n\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
                        prompt = f"{history_text}\n\n{prompt}"
                    except Exception as e:
                        logging.error(f"Error processing conversation history: {str(e)}")

                # Stream response with timeout handling
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

                # Use concurrent.futures for timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(stream_client)
                    try:
                        # Set timeout to 30 seconds
                        stream = future.result(timeout=30)
                        
                        for chunk in stream:
                            try:
                                chunk_text = chunk.choices[0].delta.content or ""
                                response += chunk_text
                                chunk_count += 1
                                
                                if chunk_count % 5 == 0:  # Stream every 5 chunks
                                    yield f"data: {json.dumps({'response': response})}\n\n"
                            except Exception as e:
                                logging.error(f"Error processing chunk: {str(e)}")
                                continue
                        
                        yield f"data: {json.dumps({'response': response})}\n\n"

                        # Save to memory
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
        data = request.json
        history = data.get('history', [])
        temp_file_path = save_chat_history(history, username)
        logging.info(f"Chat history downloaded for user: {username}")
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
    """Cancel ongoing generation process."""
    if 'username' not in session:
        return jsonify({'error': 'Please log in first.'}), 401
    
    try:
        GENERATION_CANCELLED.set()
        GENERATION_RUNNING.clear()
        logging.info(f"Generation cancelled by user: {session['username']}")
        return jsonify({'status': 'Generation cancelled'})
    except Exception as e:
        logging.error(f"Error cancelling generation: {str(e)}")
        return jsonify({'error': f'Error cancelling generation: {str(e)}'}), 500

@app.route('/api/generation_status', methods=['GET'])
def generation_status():
    """Get current generation status."""
    if 'username' not in session:
        return jsonify({'error': 'Please log in first.'}), 401
    
    try:
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

# Cleanup function for application shutdown
def cleanup_temp_files():
    """Clean up temporary files on application shutdown."""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
            logging.info(f"Cleaned up upload folder: {UPLOAD_FOLDER}")
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