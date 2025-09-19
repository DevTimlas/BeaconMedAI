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
import timeout_decorator  # Add this for timeout handling
import concurrent.futures
from functools import partial

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session key

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

def rerank_documents(query, documents, top_k=2):
    """Rerank retrieved documents based on embedding similarity."""
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

class DummyProgress:
    """Mimic Gradio's progress for Flask compatibility."""
    def __call__(self, value=None, desc=None, total=None, unit=None, track_tqdm=False):
        if desc:
            logging.info(f"Progress: {desc} ({value}/{total} {unit})")
        return self

    def tqdm(self, iterable, desc=None, total=None):
        logging.info(f"Processing {desc} with {len(iterable) if total is None else total} items")
        return iterable

async def collect_file_results(file_paths, username, progress, cancelled):
    """Collect results from process_file, handling non-async generator cases."""
    results = []
    try:
        if not file_paths:
            logging.error("No file paths provided to process_file")
            return ["Error: No files provided"]

        for file_path in file_paths:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # Size in KB
                logging.info(f"Processing file: {file_path}, size: {file_size:.2f} KB")
            else:
                logging.error(f"File does not exist: {file_path}")
                return [f"Error: File {file_path} does not exist"]

        if not callable(process_file):
            logging.error("process_file is not callable")
            return ["Error: process_file is not a function"]

        result = process_file(file_paths, username, progress or DummyProgress(), cancelled)
        if result is None:
            logging.error("process_file returned None")
            return ["Error: process_file returned None"]

        if hasattr(result, '__aiter__'):
            async for result in result:
                if result is None:
                    logging.error("process_file yielded None")
                    return ["Error: process_file yielded None"]
                if result == "Generation cancelled." or result.startswith("Error:"):
                    return [result]
                results.append(result)
        else:
            logging.error("process_file is not an async generator")
            return ["Error: process_file is not an async generator"]

        if not results:
            logging.error("process_file yielded no valid results")
            return ["Error: No valid results from file processing"]
        return results
    except Exception as e:
        logging.error(f"Error in collect_file_results: {str(e)}")
        return [f"Error: {str(e)}"]

async def collect_combine_results(reports, username, progress, cancelled):
    """Collect results from get_combine_reports, handling non-async generator cases."""
    results = []
    try:
        if not reports:
            logging.error("No reports provided to get_combine_reports")
            return ["Error: No reports provided"]
        result = get_combine_reports(reports, username, progress or DummyProgress(), cancelled)
        if result is None:
            logging.error("get_combine_reports returned None")
            return ["Error: get_combine_reports returned None"]
        if hasattr(result, '__aiter__'):
            async for result in result:
                if result is None:
                    logging.error("get_combine_reports yielded None")
                    return ["Error: get_combine_reports yielded None"]
                if result == "Generation cancelled." or result.startswith("Error:"):
                    return [result]
                results.append(result)
        else:
            logging.error("get_combine_reports is not an async generator")
            return ["Error: get_combine_reports is not an async generator"]
        if not results:
            logging.error("get_combine_reports yielded no valid results")
            return ["Error: No valid results from report combination"]
        return results
    except Exception as e:
        logging.error(f"Error in collect_combine_results: {str(e)}")
        return [f"Error: {str(e)}"]

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/api/login', methods=['POST'])
def login():
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

@app.route('/api/logout', methods=['POST'])
def logout():
    username = session.pop('username', None)
    if username in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[username]
        logging.info(f"User {username} logged out")
    return jsonify({'status': 'success', 'message': 'Logged out'})

@app.route('/api/get_username', methods=['GET'])
def get_username():
    username = session.get('username')
    if username and username in ACTIVE_SESSIONS:
        if (datetime.now() - ACTIVE_SESSIONS[username]).total_seconds() < SESSION_TIMEOUT:
            return jsonify({'username': username})
    return jsonify({'username': None})

@app.route('/api/update_vectorstore', methods=['POST'])
def update_vectorstore():
    if 'username' not in session:
        logging.error("Vectorstore update attempted without login")
        return jsonify({'status': 'error', 'message': 'Please log in first.'}), 401
    data = request.json
    edition = data.get('edition', '4th Edition')
    try:
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
        try:
            if files:
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        file_size = os.path.getsize(file_path) / 1024  # Size in KB
                        logging.info(f"Saved file: {file_path}, size: {file_size:.2f} KB")
                        file_paths.append(file_path)
                    else:
                        logging.error(f"Invalid file: {file.filename}")
                        yield f"data: {json.dumps({'report': f'Error: Invalid file {file.filename}'})}\n\n"
                        return

                partial_results = asyncio.run(collect_file_results(file_paths, username, DummyProgress(), GENERATION_CANCELLED))
                for partial_result in partial_results:
                    if partial_result == "Generation cancelled." or partial_result.startswith("Error:"):
                        GENERATION_RUNNING.clear()
                        yield f"data: {json.dumps({'report': partial_result})}\n\n"
                        return
                    all_reports.append(partial_result)
                    logging.info(f"Streaming partial report: {partial_result[:50]}...")
                    yield f"data: {json.dumps({'report': markdown.markdown(partial_result)})}\n\n"

            if text_input.strip():
                moderation_result = check_moderation(text_input)
                if moderation_result:
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'report': f'Content flagged: {moderation_result}'})}\n\n"
                    return
                text_report = asyncio.run(get_ama_impairment_report(text_input, username, DummyProgress(), GENERATION_CANCELLED, current_vectorstore))
                if GENERATION_CANCELLED.is_set():
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'report': 'Generation cancelled.'})}\n\n"
                    return
                if not files:
                    partial_results = asyncio.run(collect_combine_results([text_report], username, DummyProgress(), GENERATION_CANCELLED))
                    for combined_report in partial_results:
                        if combined_report == "Generation cancelled." or combined_report.startswith("Error:"):
                            GENERATION_RUNNING.clear()
                            yield f"data: {json.dumps({'report': combined_report})}\n\n"
                            return
                        all_reports.append(combined_report)
                        logging.info(f"Streaming combined report: {combined_report[:50]}...")
                        yield f"data: {json.dumps({'report': markdown.markdown(combined_report)})}\n\n"

            if not all_reports:
                GENERATION_RUNNING.clear()
                yield f"data: {json.dumps({'report': 'No input provided.'})}\n\n"
                return

            final_report_text = all_reports[-1]
            user_data.report_data = user_data.encrypt_data(final_report_text)
            pdf_buffer = create_pdf_from_markdown(final_report_text)
            pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
            GENERATION_RUNNING.clear()
            logging.info(f"Streaming final report: {final_report_text[:50]}...")
            yield f"data: {json.dumps({'report': markdown.markdown(final_report_text), 'pdf': pdf_base64})}\n\n"

        except Exception as e:
            logging.error(f"Error in report generation: {e}")
            GENERATION_RUNNING.clear()
            yield f"data: {json.dumps({'report': f'Error: {str(e)}'})}\n\n"
        finally:
            for file_path in file_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted uploaded file: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {str(e)}")

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/generate_rebuttal', methods=['POST'])
def generate_rebuttal():
    if 'username' not in session:
        logging.error("Rebuttal generation attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    username = session['username']
    GENERATION_RUNNING.set()
    GENERATION_CANCELLED.clear()
    temp_extract_dir = tempfile.mkdtemp()

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
        file_paths = []
        try:
            if files:
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        file_size = os.path.getsize(file_path) / 1024  # Size in KB
                        logging.info(f"Saved file: {file_path}, size: {file_size:.2f} KB")
                        file_paths.append(file_path)
                    else:
                        logging.error(f"Invalid file: {file.filename}")
                        yield f"data: {json.dumps({'rebuttal': f'Error: Invalid file {file.filename}'})}\n\n"
                        return

                partial_results = asyncio.run(collect_file_results(file_paths, username, DummyProgress(), GENERATION_CANCELLED))
                for rebuttal_text in partial_results:
                    if rebuttal_text == "Generation cancelled." or rebuttal_text.startswith("Error:"):
                        GENERATION_RUNNING.clear()
                        yield f"data: {json.dumps({'rebuttal': rebuttal_text})}\n\n"
                        return
                    all_responses.append(rebuttal_text)
                    logging.info(f"Streaming rebuttal: {rebuttal_text[:50]}...")
                    yield f"data: {json.dumps({'rebuttal': markdown.markdown(rebuttal_text)})}\n\n"

            if text_input.strip():
                moderation_result = check_moderation(text_input)
                if moderation_result:
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'rebuttal': f'Content flagged: {moderation_result}'})}\n\n"
                    return
                rebuttal_text = asyncio.run(process_batch_for_rebuttal([text_input], username, DummyProgress(), GENERATION_CANCELLED, current_vectorstore))
                if rebuttal_text == "Generation cancelled.":
                    GENERATION_RUNNING.clear()
                    yield f"data: {json.dumps({'rebuttal': 'Generation cancelled.'})}\n\n"
                    return
                all_responses.append(rebuttal_text)
                logging.info(f"Streaming text rebuttal: {rebuttal_text[:50]}...")
                yield f"data: {json.dumps({'rebuttal': markdown.markdown(rebuttal_text)})}\n\n"

            final_rebuttal_text = user_data.decrypt_data(user_data.rebuttal_data) if user_data.rebuttal_data else all_responses[-1]
            if final_rebuttal_text and final_rebuttal_text != "Generation cancelled.":
                pdf_buffer = create_pdf_from_markdown(final_rebuttal_text)
                pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
                GENERATION_RUNNING.clear()
                logging.info(f"Streaming final rebuttal: {final_rebuttal_text[:50]}...")
                yield f"data: {json.dumps({'rebuttal': markdown.markdown(final_rebuttal_text), 'pdf': pdf_base64})}\n\n"
            else:
                GENERATION_RUNNING.clear()
                yield f"data: {json.dumps({'rebuttal': final_rebuttal_text})}\n\n"

        except Exception as e:
            logging.error(f"Error in rebuttal generation: {e}")
            GENERATION_RUNNING.clear()
            yield f"data: {json.dumps({'rebuttal': f'Error: {str(e)}'})}\n\n"
        finally:
            for file_path in file_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted uploaded file: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {str(e)}")
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            logging.info(f"Cleaned up temporary directory: {temp_extract_dir}")

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/reset_reports', methods=['POST'])
def reset_reports():
    if 'username' not in session:
        logging.error("Reset reports attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    username = session['username']
    user_data = get_user_data(username)
    user_data.file_data = b""
    user_data.report_data = b""
    user_data.rebuttal_data = b""
    logging.info(f"Reports reset for user: {username}")
    return jsonify({'status': 'Reports reset'})

@app.route('/api/process_chat_files', methods=['POST'])
def process_chat_files():
    if 'username' not in session:
        logging.error("Chat file processing attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    username = session['username']
    files = request.files.getlist('files')
    user_data = get_user_data(username)
    all_pdf_content = []

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            logging.info(f"Processing chat file: {file_path}, size: {file_size:.2f} KB")
            pdf_content = get_uploaded_pdf_content(file_path, username, DummyProgress())
            if pdf_content is None:
                logging.error(f"Failed to process PDF: {file_path}")
                continue
            all_pdf_content.append(pdf_content)
            os.remove(file_path)
            logging.info(f"Deleted chat file: {file_path}")

    if all_pdf_content:
        merged_content = "\n\n".join(all_pdf_content)
        user_data.file_data = user_data.encrypt_data(merged_content)
        logging.info(f"Stored {len(all_pdf_content)} PDF contents for user: {username}")
    return jsonify({'status': 'Files processed'})

@app.route('/api/handle_query', methods=['POST'])
def handle_query():
    if 'username' not in session:
        logging.error("Query handling attempted without login")
        return jsonify({'response': 'Please log in first.'}), 401
    username = session['username']
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
            if user_data.file_data:
                prompt += f"File content:\n{user_data.decrypt_data(user_data.file_data)}\n\n"
            if user_data.report_data:
                prompt += f"Generated report:\n{user_data.decrypt_data(user_data.report_data)}\n\n"
            if user_data.rebuttal_data:
                prompt += f"Generated rebuttal:\n{user_data.decrypt_data(user_data.rebuttal_data)}\n\n"

            # Retrieve relevant AMA content
            if current_vectorstore:
                retrieved_docs = current_vectorstore.similarity_search(query, k=2)
                retrieved_docs = rerank_documents(query, retrieved_docs, top_k=2)
                retrieved_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                if retrieved_content:
                    prompt += f"Relevant AMA content:\n{retrieved_content}\n\n"

            prompt += """
                - If no rebuttal data, inform user to generate one if relevant.
                - If no report data, inform user to generate one if relevant.
                - If unrelated to AMA, guide user back to task.
                - Be specific and precise, provide AMA references for medical queries.
            """

            # Add conversation history
            if history:
                history_text = "\n\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
                prompt = f"{history_text}\n\n{prompt}"

            # Stream response with timeout handling
            response = ""
            chunk_count = 0

            def stream_client():
                return init_client(
                    prompt,
                    model_pro='gpt',  # Adjust as needed
                    max_tokens=1000,
                    stream=True
                )

            # Use concurrent.futures for timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(stream_client)
                try:
                    # Set timeout to 30 seconds
                    stream = future.result(timeout=30)
                    for chunk in stream:
                        chunk_text = chunk.choices[0].delta.content or ""
                        response += chunk_text
                        chunk_count += 1
                        if chunk_count % 5 == 0:  # Stream every 5 chunks
                            yield f"data: {json.dumps({'response': response})}\n\n"
                    yield f"data: {json.dumps({'response': response})}\n\n"

                    # Save to memory
                    user_data.memory.save_context({"input": query}, {"output": response})

                except concurrent.futures.TimeoutError:
                    logging.error("Timeout in stream_client call")
                    yield f"data: {json.dumps({'response': 'Error: Request to language model timed out. Please try again.'})}\n\n"

        except Exception as e:
            logging.error(f"Error in handle_query: {str(e)}")
            yield f"data: {json.dumps({'response': f'Error: {str(e)}'})}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/api/download_chat_history', methods=['POST'])
def download_chat_history_route():
    if 'username' not in session:
        logging.error("Chat history download attempted without login")
        return jsonify({'error': 'Please log in first.'}), 401
    username = session['username']
    data = request.json
    history = data.get('history', [])
    temp_file_path = save_chat_history(history, username)
    logging.info(f"Chat history downloaded for user: {username}")
    return send_file(temp_file_path, as_attachment=True, download_name='chat_history.txt')

@app.route('/favicon.ico')
def favicon():
    return send_file('favicon.ico', mimetype='image/x-icon')

if __name__ == '__main__':
    initialize_vectorstore_for_edition("4th Edition")
    app.run(debug=True)