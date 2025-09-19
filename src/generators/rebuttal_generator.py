import sys
import os
import asyncio
from langchain.schema import HumanMessage, AIMessage
from openai import OpenAI

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.user_data import get_user_data
from core.vectorstore import get_ensemble_retriever, rerank_documents
from utils.web_utils import web_search_url, summarize_content
from utils.logging_utils import logging
from core.models import init_client


async def process_batch_for_rebuttal(page_texts, username, progress, generation_cancelled, vectorstore):
    """
    Process a batch of page texts for rebuttal generation.

    Args:
        page_texts: List of page content strings
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event to check for cancellation
        vectorstore: Chroma vectorstore for AMA content

    Returns:
        Generated rebuttal text or cancellation message
    """
    if generation_cancelled.is_set():
        progress(1.0, desc="Processing file")
        return "Generation cancelled."
    progress(0.3, desc="Getting report")
    combined_text = " ".join(page_texts)

    user_data = get_user_data(username)
    conversation_history = user_data.memory.load_memory_variables({})["history"]
    similarity_threshold = 0.6
    for i in range(0, len(conversation_history), 2):
        if i + 1 >= len(conversation_history):
            continue
        human_message = conversation_history[i]
        ai_message = conversation_history[i + 1]
        if isinstance(human_message, HumanMessage) and isinstance(ai_message, AIMessage):
            previous_input = human_message.content
            from processors.text_processors import check_text_similarity
            similarity = check_text_similarity(combined_text, previous_input, method='cosine')
            logging.info(f"Similarity with previous input {i // 2 + 1}: {similarity}")
            if similarity >= similarity_threshold:
                logging.info(f"Returning stored response for similar input (similarity: {similarity})")
                return user_data.decrypt_data(ai_message.content) if isinstance(ai_message.content,
                                                                                bytes) else ai_message.content

    summ = summarize_content(combined_text, "gpt-4o-mini")
    web_searches = web_search_url(summ, 'gpt')

    unique_words = len(set(summ.split()))
    k = min(10, max(5, unique_words // 50))
    fetch_k = 5 * k
    lambda_mult = 0.8
    search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}

    ensemble_retriever = get_ensemble_retriever(summ, k, fetch_k, lambda_mult)
    retrieved_docs = ensemble_retriever.get_relevant_documents(summ)
    reranked_docs = rerank_documents(summ, retrieved_docs, top_k=k)
    retrieved_content = "\n\n".join([doc.page_content for doc in reranked_docs])

    from core.models import api_key
    mod_client = OpenAI(api_key=api_key)
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: mod_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal document analysis assistant. Your task is to extract and summarize key elements from legal documents for rebuttal preparation."
                },
                {
                    "role": "user",
                    "content": f"""Analyze this legal document and provide a structured summary:

                    {combined_text}

                    **Relevant AMA Content:**
                    {retrieved_content}

                    Follow these requirements:
                    1. Extract key identifying details:
                       - Court File Number
                       - Court Name
                       - Parties (Plaintiff/Defendant)
                       - Document Type
                       - Date of document
                       - Opposing expert's name/credentials (if present)

                    2. Identify 5-10 significant claims/arguments
                       - For each claim, note:
                         * Page number reference
                         * Apparent weaknesses/contradictions
                         * Potential counter-evidence sources

                    3. Format as concise bullet points (500-800 words)

                    Important:
                    - Be objective
                    - Preserve page references
                    - Highlight missing information
                    - Note if document appears incomplete
                    - Online references {web_searches}
                    - DO NOT MAKE UP ANYTHING"""
                }
            ],
            temperature=0.3,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
    )
    response_text = response.choices[0].message.content.strip() + web_searches
    user_data.memory.save_context({"input": combined_text}, {"output": user_data.encrypt_data(response_text)})
    progress(0.7, desc="Getting report")
    return response_text