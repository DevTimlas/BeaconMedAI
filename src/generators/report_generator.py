import sys
import os
import asyncio
from langchain.schema import HumanMessage, AIMessage

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.user_data import get_user_data
from core.vectorstore import get_ensemble_retriever, rerank_documents
from utils.helpers import estimate_tokens, get_metadata_filter
from utils.web_utils import web_search_url, summarize_content, no_summarize_content
from utils.logging_utils import logging
from core.models import init_client


async def process_ama_report_batch(pages, username, progress, generation_cancelled):
    """
    Process a batch of PDF pages to generate an AMA impairment report.

    Args:
        pages: List of page content strings
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event to check for cancellation

    Returns:
        Encrypted report text or cancellation message
    """
    progress(0.2, desc="Getting report")
    logging.info(f"Processing batch for user {username}")
    combined_text = " ".join(pages)
    response = await get_ama_impairment_report(combined_text, username, progress, generation_cancelled)
    progress(0.8, desc="Getting report")
    await asyncio.sleep(0)
    user_data = get_user_data(username)
    return user_data.encrypt_data(response)


async def get_ama_impairment_report(bot_report, username, progress, generation_cancelled, vectorstore):
    """
    Generate an AMA impairment report based on input text.

    Args:
        bot_report: Input report text
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event to check for cancellation
        vectorstore: Chroma vectorstore for AMA content

    Returns:
        Generated report text
    """
    progress(0.05, desc="Getting report")
    logging.info(f"Generating impairment report for user {username}")
    user_data = get_user_data(username)
    await asyncio.sleep(0)

    # Check conversation history for similar inputs
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
            similarity = check_text_similarity(bot_report, previous_input, method='cosine')
            logging.info(f"Similarity with previous input {i // 2 + 1}: {similarity}")
            if similarity >= similarity_threshold:
                logging.info(f"Returning stored response for similar input (similarity: {similarity})")
                return user_data.decrypt_data(ai_message.content) if isinstance(ai_message.content,
                                                                                bytes) else ai_message.content

    progress(0.1, desc="Getting report")
    bot_report_sum = summarize_content(bot_report) if len(bot_report.split()) > 2000 else bot_report
    bot_report_reconstructed = no_summarize_content(bot_report)
    await asyncio.sleep(0)
    print('original report', bot_report)
    print('summarized report', bot_report_sum)
    print('reconstructed report', bot_report_reconstructed)

    progress(0.2, desc="Getting report")
    unique_words = len(set(bot_report_sum.split()))
    k = min(10, max(3, unique_words // 30))
    fetch_k = 5 * k
    lambda_mult = 0.5
    search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}

    progress(0.3, desc="Getting report")
    metadata_filter = get_metadata_filter(bot_report_reconstructed, vectorstore, max_terms=20)
    if metadata_filter and "chapter" in metadata_filter:
        search_kwargs["filter"] = {"chapter": {"$in": metadata_filter["chapter"]}}
        query = bot_report_sum
        logging.info(f"Using chapter-based filter: {metadata_filter['chapter']}")
    elif metadata_filter and "terms" in metadata_filter:
        terms = metadata_filter["terms"]
        query = " ".join(terms)
        logging.info(f"Using term-based query: {query}")
    else:
        query = bot_report_sum
        logging.info("No filter applied, using original report text")
    logging.info(
        f"Retrieving AMA content with k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult} based on {unique_words} unique words")

    ensemble_retriever = get_ensemble_retriever(query, k, fetch_k, lambda_mult, metadata_filter)
    retrieved_docs = ensemble_retriever.get_relevant_documents(query)
    logging.info(f"Initially retrieved {len(retrieved_docs)} documents")

    progress(0.35, desc="Getting report")
    reranked_docs = rerank_documents(query, retrieved_docs, top_k=k)
    logging.info(f"Reranked to {len(reranked_docs)} documents")
    retrieved_content = "\n\n".join([doc.page_content for doc in reranked_docs])
    await asyncio.sleep(0)

    progress(0.4, desc="Getting report")
    web_searches = web_search_url(bot_report_reconstructed, 'gpt')
    await asyncio.sleep(0)

    progress(0.5, desc="Getting report")
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
    prompt_file = os.path.join(template_dir, 'ama_imp_prmpt1.txt')
    with open(prompt_file, encoding='utf-8') as f:
        ama_prompt = f.read()
    add_prompt = f"""
        **AMA Reference Content:**
        {retrieved_content}

        **Main Content Summary:**
        {bot_report_sum}   

        **Medical related terms to focus on and find their classes and points:**
        {bot_report_reconstructed}

        **Also make sure points and classes are accurate when calculating the impairment.**
        **Always replace placeholders e.g X with the actual values.**
        **Ensure you increase your accuracy on the WPI calculation section, as it's the most important part of the report, so prioritize it.**
    """
    prompt = ama_prompt + add_prompt + web_searches
    await asyncio.sleep(0)

    progress(0.6, desc="Getting report")
    response = ""
    # for chunk in init_client(prompt, model_pro='groq', model_type='llama-3.3-70b-versatile', max_tokens=5000, stream=True):
    for chunk in init_client(prompt, model_pro='gpt', max_tokens=5000, stream=True):
        if generation_cancelled.is_set():
            progress(1.0, desc="Processing file")
            logging.info(f"Generation cancelled during report generation for user {username}")
            return "Generation cancelled."
        response += chunk.choices[0].delta.content or ""
    logging.info(f"Generated impairment report for user {username}")

    progress(0.9, desc="Getting report")
    user_data.memory.save_context({"input": bot_report}, {"output": user_data.encrypt_data(response)})
    progress(1.0, desc="Getting report")
    await asyncio.sleep(0)
    return response