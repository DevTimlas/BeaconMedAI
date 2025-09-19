import sys
import os
import gradio as gr
import asyncio
import threading
import tempfile
import time
import markdown
import shutil
from datetime import datetime
from gradio_modal import Modal
from gradio_pdf import PDF

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Global variables
AUTH_CREDENTIALS = {
    "michael rice": "4321",
    "Tim": "1038",
    "Tester": "Test012"
}
username_state = None
GENERATION_RUNNING = threading.Event()
GENERATION_CANCELLED = threading.Event()


def gradio_ui():
    """
    Create and configure the Gradio interface for BeaconMedicalAi AMA.

    Returns:
        Gradio Blocks interface
    """
    global username_state

    async def generate_and_display_report(file_paths, txt_input, username, edition):
        global GENERATION_RUNNING, GENERATION_CANCELLED
        GENERATION_RUNNING.set()
        GENERATION_CANCELLED.clear()

        if not username:
            gr.Warning("Please log in first.")
            GENERATION_RUNNING.clear()
            yield (
                gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                          value="Generate Beacon AMA Report", variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False),
                gr.update(visible=True)
            )
            return

        initialize_vectorstore_for_edition(edition)
        user_data = get_user_data(username)
        yield (
            gr.update(visible=False),
            gr.update(visible=True, elem_classes=["custom-btn", "stop-btn"]),
            gr.update(visible=True, value="Processing file"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
        await asyncio.sleep(0.01)

        try:
            if not file_paths and not txt_input:
                gr.Warning("Please upload at least one document or provide text input.")
                GENERATION_RUNNING.clear()
                yield (
                    gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                              value="Generate Beacon AMA Report", variant="primary"),
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(value="Please provide input.", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
                return

            final_report_text = None
            pdf_buffer = None
            all_reports = []

            if file_paths:
                progress = gr.Progress(track_tqdm=True)
                progress(0.0, desc="Processing file")
                async for partial_result in process_file(file_paths, username, progress, GENERATION_CANCELLED):
                    if partial_result == "Generation cancelled.":
                        GENERATION_RUNNING.clear()
                        yield (
                            gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                                      value="Generate Beacon AMA Report", variant="primary"),
                            gr.update(visible=False),
                            gr.update(visible=False, value=""),
                            gr.update(value="Generation cancelled.", visible=True),
                            gr.update(visible=False),
                            gr.update(visible=True)
                        )
                        return
                    all_reports.append(partial_result)
                    yield (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True, value="Getting report"),
                        gr.update(value=markdown.markdown(partial_result), visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                    await asyncio.sleep(0.01)

            if txt_input.strip():
                moderation_result = check_moderation(txt_input)
                if moderation_result:
                    GENERATION_RUNNING.clear()
                    yield (
                        gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                                  value="Generate Beacon AMA Report", variant="primary"),
                        gr.update(visible=False),
                        gr.update(visible=False, value=""),
                        gr.update(value=f"Content flagged: {moderation_result}", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True)
                    )
                    return
                progress = gr.Progress(track_tqdm=True)
                progress(0.0, desc="Processing text")
                text_report = await get_ama_impairment_report(txt_input, username, progress, GENERATION_CANCELLED,
                                                              current_vectorstore)
                if GENERATION_CANCELLED.is_set():
                    GENERATION_RUNNING.clear()
                    yield (
                        gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                                  value="Generate Beacon AMA Report", variant="primary"),
                        gr.update(visible=False),
                        gr.update(visible=False, value=""),
                        gr.update(value="Generation cancelled.", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True)
                    )
                    return
                if not file_paths:
                    async for combined_report in get_combine_reports([text_report], username, progress,
                                                                     GENERATION_CANCELLED):
                        if combined_report == "Generation cancelled.":
                            GENERATION_RUNNING.clear()
                            yield (
                                gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                                          value="Generate Beacon AMA Report", variant="primary"),
                                gr.update(visible=False),
                                gr.update(visible=False, value=""),
                                gr.update(value="Generation cancelled.", visible=True),
                                gr.update(visible=False),
                                gr.update(visible=True)
                            )
                            return
                        all_reports.append(combined_report)
                        yield (
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=True, value="Getting report"),
                            gr.update(value=markdown.markdown(combined_report), visible=True),
                            gr.update(visible=False),
                            gr.update(visible=False)
                        )
                        await asyncio.sleep(0.01)

            if not all_reports:
                GENERATION_RUNNING.clear()
                yield (
                    gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                              value="Generate Beacon AMA Report", variant="primary"),
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(value="No input provided.", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
                return

            final_report_text = all_reports[-1]
            user_data.report_data = user_data.encrypt_data(final_report_text)

            progress(0.95, desc="Generating PDF")
            pdf_buffer = create_pdf_from_markdown(final_report_text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_buffer.getvalue())
                tmp_file_path = tmp_file.name

            GENERATION_RUNNING.clear()
            yield (
                gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                          value="Generate Beacon AMA Report", variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(value=markdown.markdown(final_report_text), visible=True),
                gr.update(value=tmp_file_path, label="Download Report", visible=True),
                gr.update(visible=True)
            )

        except Exception as e:
            logging.error(f"Error in report generation: {e}")
            GENERATION_RUNNING.clear()
            yield (
                gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                          value="Generate Beacon AMA Report", variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(value=f"Error: {str(e)}", visible=True),
                gr.update(visible=False),
                gr.update(visible=True)
            )

    async def generate_and_display_rebuttals(file_paths, txt_input, username, edition):
        global GENERATION_RUNNING, GENERATION_CANCELLED
        GENERATION_RUNNING.set()
        GENERATION_CANCELLED.clear()
        temp_extract_dir = tempfile.mkdtemp()

        if not username:
            gr.Warning("Please log in first.")
            GENERATION_RUNNING.clear()
            yield (
                gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                          value="Generate Beacon AMA Rebuttal", variant="secondary"),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(visible=False, value=""),
                gr.update(visible=False),
                gr.update(visible=True)
            )
            return

        initialize_vectorstore_for_edition(edition)
        user_data = get_user_data(username)
        yield (
            gr.update(visible=False),
            gr.update(visible=True, elem_classes=["custom-btn", "stop-btn"]),
            gr.update(visible=True, value="Processing file"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
        await asyncio.sleep(0)

        try:
            if not file_paths and not txt_input.strip():
                gr.Warning("Please upload a document or provide text.")
                GENERATION_RUNNING.clear()
                yield (
                    gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                              value="Generate Beacon AMA Rebuttal", variant="secondary"),
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(value="Please upload a file or enter text.", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
                return

            all_responses = []
            progress = gr.Progress(track_tqdm=True)

            if file_paths:
                async for rebuttal_text in process_file(file_paths, username, progress, GENERATION_CANCELLED):
                    if rebuttal_text == "Generation cancelled." or "Error:" in rebuttal_text:
                        GENERATION_RUNNING.clear()
                        yield (
                            gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                                      value="Generate Beacon AMA Rebuttal", variant="secondary"),
                            gr.update(visible=False),
                            gr.update(visible=False, value=""),
                            gr.update(value=rebuttal_text, visible=True),
                            gr.update(visible=False),
                            gr.update(visible=True)
                        )
                        return
                    all_responses.append(rebuttal_text)
                    yield (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True, value="Getting rebuttal"),
                        gr.update(value=markdown.markdown(rebuttal_text), visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
                    await asyncio.sleep(0.01)

            if txt_input.strip():
                moderation_result = check_moderation(txt_input)
                if moderation_result:
                    GENERATION_RUNNING.clear()
                    yield (
                        gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                                  value="Generate Beacon AMA Rebuttal", variant="secondary"),
                        gr.update(visible=False),
                        gr.update(visible=False, value=""),
                        gr.update(value=f"Content flagged: {moderation_result}", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True)
                    )
                    return
                rebuttal_text = await process_batch_for_rebuttal([txt_input], username, progress, GENERATION_CANCELLED,
                                                                 current_vectorstore)
                if rebuttal_text == "Generation cancelled.":
                    GENERATION_RUNNING.clear()
                    yield (
                        gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                                  value="Generate Beacon AMA Rebuttal", variant="secondary"),
                        gr.update(visible=False),
                        gr.update(visible=False, value=""),
                        gr.update(value="Generation cancelled.", visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True)
                    )
                    return
                all_responses.append(rebuttal_text)
                yield (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True, value="Getting rebuttal"),
                    gr.update(value=markdown.markdown(rebuttal_text), visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
                await asyncio.sleep(0.01)

            final_rebuttal_text = user_data.decrypt_data(user_data.rebuttal_data) if user_data.rebuttal_data else \
                all_responses[-1]

            if final_rebuttal_text and final_rebuttal_text != "Generation cancelled.":
                progress(0.95, desc="Generating PDF")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    pdf_buffer = create_pdf_from_markdown(final_rebuttal_text)
                    tmp_file.write(pdf_buffer.getvalue())
                    tmp_file_path = tmp_file.name

                GENERATION_RUNNING.clear()
                yield (
                    gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                              value="Generate Beacon AMA Rebuttal", variant="secondary"),
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(value=markdown.markdown(final_rebuttal_text), visible=True),
                    gr.update(value=tmp_file_path, label="Download Rebuttal", visible=True),
                    gr.update(visible=True)
                )
            else:
                GENERATION_RUNNING.clear()
                yield (
                    gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                              value="Generate Beacon AMA Rebuttal", variant="secondary"),
                    gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    gr.update(value=final_rebuttal_text, visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                )

        except Exception as e:
            logging.error(f"Error in rebuttal generation: {e}")
            GENERATION_RUNNING.clear()
            yield (
                gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"],
                          value="Generate Beacon AMA Rebuttal", variant="secondary"),
                gr.update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(value=f"Error: {str(e)}", visible=True),
                gr.update(visible=False),
                gr.update(visible=True)
            )
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

    def stop_generation():
        global GENERATION_RUNNING, GENERATION_CANCELLED
        if GENERATION_RUNNING.is_set():
            GENERATION_CANCELLED.set()
            logging.info("Stop generation requested")
            return gr.update(value="Cancellation requested...", visible=True)
        return gr.update(visible=False)

    def update_preview(file_paths):
        if not file_paths:
            logging.info("No files provided for preview")
            return gr.update(visible=False), gr.update(visible=False)
        image_files = [f for f in file_paths if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        pdf_file = next((f for f in file_paths if f.lower().endswith(".pdf")), None)
        logging.info(f"Previewing {len(image_files)} images and {'1 PDF' if pdf_file else 'no PDF'}")
        return (
            gr.update(value=image_files, visible=bool(image_files)),
            gr.update(value=pdf_file, visible=bool(pdf_file))
        )

    def reset_reports(username):
        if not username:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        user_data = get_user_data(username)
        user_data.file_data = b""
        user_data.report_data = b""
        user_data.rebuttal_data = b""
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    def toggle_chatbot(chatbot_visible, file_paths, username):
        if not username:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        if chatbot_visible:
            if file_paths:
                all_pdf_content = []
                for file_path in file_paths:
                    if file_path.lower().endswith(".pdf"):
                        pdf_content = get_uploaded_pdf_content(file_path, username, gr.Progress())
                        all_pdf_content.append(pdf_content)
                merged_content = "\n\n".join(all_pdf_content)
                user_data = get_user_data(username)
                user_data.file_data = user_data.encrypt_data(merged_content)
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def toggle_mode(mode):
        if mode == "BeaconMedicalAi Report mode":
            return (
                gr.update(visible=True, elem_classes=["custom-btn", "BeaconMedicalAi-btn"],
                          value="Generate Beacon AMA Report", variant="primary"),
                gr.update(visible=False),
                gr.Info("Beacon Medical Ai AMA Guides to Evaluation of PERMANENT Impairment Report", duration=5,
                        title='REPORT MODE')
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, elem_classes=["custom-btn", "rebuttal-btn"], value="Generate Beacon AMA Rebuttal",
                      variant="secondary"),
            gr.Info("Beacon Medical Ai AMA Guides to Evaluation of PERMANENT Impairment Rebuttal", duration=5,
                    title='REBUTTAL MODE')
        )

    def update_vectorstore(edition):
        try:
            initialize_vectorstore_for_edition(edition)
            return gr.update(value=f"Using AMA Guides {edition}")
        except Exception as e:
            logging.error(f"Error updating vectorstore: {e}")
            return gr.update(value=f"Error: Failed to load {edition} vectorstore")

    def handle_query(query, history, username):
        if not username:
            return history, "Please log in first."
        if not query:
            return history, "Please enter a message."
        user_data = get_user_data(username)
        moderation_result = check_moderation(query)
        if moderation_result:
            return history, f"Content flagged: {moderation_result}"
        from core.models import init_client
        prompt = f"""
        You are BeaconMedicalAi AMA, a medical-legal assistant. Answer the following question based on the user's chat history and AMA Guides:
        **Question**: {query}
        **Chat History**: {history}
        """
        response = init_client(prompt, model_pro='groq', model_type='llama-3.3-70b-versatile', max_tokens=1000)
        user_data.memory.save_context({"input": query}, {"output": user_data.encrypt_data(response)})
        history.append([query, response])
        return history, ""

    def update_chat_history(chatbot_history, username):
        if not username:
            return gr.update(visible=False)
        temp_file_path = save_chat_history(chatbot_history, username)
        return gr.update(value=temp_file_path, visible=True, label="Download Chat History")

    with gr.Blocks(title="BeaconMedicalAi AMA", fill_width=True, css_paths="style.css",
                   theme=gr.themes.Default(primary_hue='teal')) as demo:
        username_state = gr.State()
        user_name = gr.Markdown(value="Not logged in", elem_classes=["welcome-msg"])

        with gr.Row(elem_classes=["top-bar"]):
            clear_button = gr.Button("Clear report", variant="stop", visible=False,
                                     elem_classes=["custom-btn", "clear-btn"])
            download_button = gr.DownloadButton(label="Download PDF Report", visible=False,
                                                elem_classes=["custom-btn", "download-btn"])

        with gr.Row():
            with gr.Sidebar():
                gr.Markdown("### Upload and Preview Medical Files")
                file_input = gr.File(
                    label="Upload File (ZIP, PDF, Audio, Video, or Image)",
                    file_types=[".zip", ".pdf", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".jpg", ".jpeg",
                                ".png"],
                    file_count="multiple",
                    height=150,
                    allow_reordering=True
                )
                with gr.Accordion("Text Input", open=False):
                    text_input = gr.Textbox(
                        label="Or paste text here",
                        placeholder="Type or paste your input text...",
                        lines=6,
                        max_lines=20,
                        info="No file upload? Simply input text here to generate report or rebuttal"
                    )
                with gr.Row(elem_classes=["preview-row"]):
                    preview_button = gr.Button("Preview Files", variant="primary",
                                               elem_classes=["custom-btn", "preview-btn"])
                    edition_dropdown = gr.Dropdown(
                        choices=["4th Edition", "6th Edition"],
                        value="4th Edition",
                        label="AMA Guides Edition",
                        info="Select the AMA Guides edition to use"
                    )
                edition_status = gr.Markdown(value="Using AMA Guides 4th Edition")
                mode_selector = gr.Radio(
                    choices=["BeaconMedicalAi Report mode", "BeaconMedicalAi Rebuttal mode"],
                    value="BeaconMedicalAi Report mode",
                    label="Select Mode",
                    info="Toggle to switch between report and rebuttal modes"
                )
                with gr.Row(elem_classes=["center-buttons"]):
                    generate_button = gr.Button("Generate Beacon AMA Report", variant="primary", visible=True,
                                                elem_classes=["custom-btn", "BeaconMedicalAi-btn"])
                    stop_button = gr.Button("Stop Generation", variant="stop", visible=False,
                                            elem_classes=["custom-btn", "stop-btn"])
                    debate_button = gr.Button("Generate Beacon AMA Rebuttal", variant="secondary", visible=False,
                                              elem_classes=["custom-btn", "rebuttal-btn"])
                generating_label = gr.Label(value="Generating Report...", visible=False)
                chatbot_toggle = gr.Checkbox(label="Enable Chatbot", value=False)
                chat_download_button = gr.DownloadButton(label="Download Chat", visible=False,
                                                         elem_classes=["custom-btn", "download-btn"])

            with gr.Column(visible=True) as report_column:
                report_text = gr.Markdown(label="Report Preview", visible=False, height=600)

            with gr.Column(visible=False) as chatbot_column:
                chat_interface = gr.ChatInterface(
                    fn=handle_query,
                    additional_inputs=[username_state],
                    title="Chat with AMA",
                    examples=[
                        ["What does this report mean?"],
                        ["Can you explain this impairment?"],
                        ["What is the rebuttal for this argument?"]
                    ]
                )

        with Modal(visible=False, elem_id="preview-modal", elem_classes=['gradio-modal'],
                   allow_user_close=True) as preview_modal:
            gr.Markdown("### File Preview")
            image_gallery = gr.Gallery(label="Image Preview", visible=False)
            pdf_preview = PDF(label="PDF Preview", visible=False, height=500, container=False)
            close_button = gr.Button("Close", variant="stop", elem_classes=["custom-btn", "stop-btn"])

        initialize_vectorstore_for_edition("4th Edition")

        file_input.change(
            fn=update_preview,
            inputs=[file_input],
            outputs=[image_gallery, pdf_preview]
        )

        preview_button.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[preview_modal]
        )

        close_button.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[preview_modal]
        )

        edition_dropdown.change(
            fn=update_vectorstore,
            inputs=[edition_dropdown],
            outputs=[edition_status]
        )

        mode_selector.change(
            fn=toggle_mode,
            inputs=[mode_selector],
            outputs=[generate_button, debate_button, gr.Markdown()]
        )

        generate_button.click(
            fn=generate_and_display_report,
            inputs=[file_input, text_input, username_state, edition_dropdown],
            outputs=[
                generate_button,
                stop_button,
                generating_label,
                report_text,
                download_button,
                clear_button
            ],
            concurrency_limit=1
        )

        debate_button.click(
            fn=generate_and_display_rebuttals,
            inputs=[file_input, text_input, username_state, edition_dropdown],
            outputs=[
                debate_button,
                stop_button,
                generating_label,
                report_text,
                download_button,
                clear_button
            ],
            concurrency_limit=1
        )

        stop_button.click(
            fn=stop_generation,
            inputs=[],
            outputs=[generating_label],
            trigger_mode='once'
        )

        clear_button.click(
            fn=reset_reports,
            inputs=[username_state],
            outputs=[
                report_text,
                download_button,
                generating_label,
                clear_button
            ]
        )

        chatbot_toggle.change(
            fn=toggle_chatbot,
            inputs=[chatbot_toggle, file_input, username_state],
            outputs=[chatbot_column, report_column, chat_download_button]
        )

        chat_interface.chatbot.change(
            fn=update_chat_history,
            inputs=[chat_interface.chatbot, username_state],
            outputs=[chat_download_button]
        )

        chat_download_button.click(
            fn=download_chat_history,
            inputs=[username_state],
            outputs=[chat_download_button]
        )

        def set_username(request: gr.Request):
            if request.username:
                username_state.value = request.username
                return gr.update(
                    value=f"## Welcome {str(request.username).upper()} >> Chat with bot for more assistance.",
                    elem_classes=["welcome-msg"])
            return gr.update(value="Not logged in", elem_classes=["welcome-msg"])

        demo.load(set_username, None, [user_name])

    return demo
