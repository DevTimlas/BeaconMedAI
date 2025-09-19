import sys
import os
import asyncio
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.user_data import get_user_data
from utils.logging_utils import logging
from core.models import init_client


async def get_combine_reports(report_list, username, progress, generation_cancelled):
    """
    Combine multiple AMA impairment reports into a single report.

    Args:
        report_list: List of report texts (encrypted or plain)
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event to check for cancellation

    Yields:
        Combined report text or cancellation message
    """
    try:
        progress(0.1, desc="Getting report")
        logging.info(f"Combining reports for user {username}")
        user_data = get_user_data(username)

        input_text = " ".join(user_data.decrypt_data(r) if isinstance(r, bytes) else str(r) for r in report_list)
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
                similarity = check_text_similarity(input_text, previous_input, method='cosine')
                logging.info(f"Similarity with previous input {i // 2 + 1}: {similarity}")
                if similarity >= similarity_threshold:
                    logging.info(f"Returning stored response for similar input (similarity: {similarity})")
                    yield user_data.decrypt_data(ai_message.content) if isinstance(ai_message.content,
                                                                                   bytes) else ai_message.content
                    return

        progress(0.2, desc="Getting report")
        template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
        prompt_file = os.path.join(template_dir, 'ama_combine_impairment.txt')
        with open(prompt_file, encoding='utf-8') as f:
            comb_prompt = f.read()
        add_prompt = f"""
            **Source Data:**
            {input_text}

            **Online Annotations (5-10 urls)**

            **Signature**:
            Dr. {user_data.username}
            {datetime.today().strftime('%d-%b-%Y')}

            Always replace placeholders e.g X with the actual values.

            Lastly NOTE that, the final returned output is supposed to be a fully structured markdown 
                with no additional tags like ```markdown that would distort when processing it later
                and with necessary bullet points, next line spacing, well formatted and separated, do not jampack outputs.
            - Ensure you increase your accuracy on the WPI calculation section, as it's the most important part of the report, so prioritize it. also it should be in the range of 0% - 100% to avoid errors and being too specific.
            - Ensure output is a well formatted markdown, do not put things that are supposed to be key points in a straight line. separation and clear markdown output is important 
        """
        prompt = comb_prompt + add_prompt
        await asyncio.sleep(0)

        progress(0.3, desc="Getting report")
        response = ""
        # for chunk in init_client(prompt, model_pro='groq', model_type='llama-3.3-70b-versatile', max_tokens=8000, stream=True):
        for chunk in init_client(prompt, model_pro='gpt', max_tokens=8000, stream=True):
            if generation_cancelled.is_set():
                progress(1.0, desc="Processing file")
                logging.info(f"Generation cancelled during combined report generation for user {username}")
                yield "Generation cancelled."
                return
            response += chunk.choices[0].delta.content or ""
            yield response
            await asyncio.sleep(0.01)
        logging.info(f"Generated combined report for user {username}")

        progress(0.9, desc="Getting report")
        user_data.memory.save_context({"input": input_text}, {"output": user_data.encrypt_data(response)})
        user_data.report_data = user_data.encrypt_data(response)
        progress(1.0, desc="Getting report")
        await asyncio.sleep(0)
        yield response
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Error occurred whilst combining report: {e}")
        yield f"Error: {str(e)}"


async def get_combine_rebuttal(report_list, username, progress, generation_cancelled):
    """
    Combine multiple rebuttal reports into a single rebuttal document.

    Args:
        report_list: List of rebuttal texts (encrypted or plain)
        username: User's username
        progress: Gradio progress object
        generation_cancelled: Threading Event to check for cancellation

    Yields:
        Combined rebuttal text or cancellation message
    """
    if generation_cancelled.is_set():
        progress(1.0, desc="Processing file")
        yield "Generation cancelled."
        return
    user_data = get_user_data(username)
    progress(0.1, desc="Getting report")
    logging.info(f"Combining rebuttal reports for user {username}")

    input_text = " ".join(user_data.decrypt_data(r) if isinstance(r, bytes) else str(r) for r in report_list)
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
            similarity = check_text_similarity(input_text, previous_input, method='cosine')
            logging.info(f"Similarity with previous input {i // 2 + 1}: {similarity}")
            if similarity >= similarity_threshold:
                logging.info(f"Returning stored response for similar input (similarity: {similarity})")
                yield user_data.decrypt_data(ai_message.content) if isinstance(ai_message.content,
                                                                               bytes) else ai_message.content
                return

    prompt = (
        f"""You are BeaconMedicalAi AMA, a medical-legal expert preparing a formal rebuttal report. Create a comprehensive 4000+ word rebuttal document using these guidelines:

        <h1 style="text-align: center;">BeaconMedicalAi Rebuttal Report</h1>

        ## **Title & Identifying Information**
        - **Court File Number**: [From source document]
        - **Court Name**: [From source document]
        - **Parties**: Plaintiff: [Name], Defendant: [Name]
        - **Rebuttal Target**: [Document Type] by [Opposing Expert], dated [Date]
        - **Prepared for**: User ID {user_data.user_id}
        - **Date**: {datetime.today().strftime('%d-%b-%Y')}

        ## **Introduction**
        - State purpose
        - Include your credentials
        - Outline rebuttal methodology

        ## **Summary of Opposing Arguments**
        - Present opponent's key claims
        - Organize by medical topic/legal argument
        - Weaknesses/Contradictions

        ## **Point-by-Point Rebuttal** (Minimum 4000 words)
        For each opposing claim:
        1. **Claim**: [Detailed with Direct quote]
        2. **Analysis**: Methodological flaws, factual inaccuracies, omitted evidence
        3. **Counter Evidence**: Peer-reviewed studies, clinical guidelines, case precedents
        4. **Conclusion**: Why claim is invalid

        ## **Supporting Evidence** (1-2 pages)
        - Tables comparing studies
        - Diagnostic data analysis
        - Expert consensus statements

        ## **Conclusion** (1 page)
        - Synthesis of rebuttal findings
        - Professional opinion
        - Recommendations

        ## **References** (including up to 5 urls from online searches)

        ## **Signature**
        Dr. {user_data.username}
        {datetime.today().strftime('%d-%b-%Y')}

        **Source Materials:**
        {input_text}

        **Formatting Rules:**
        - APA citation style
        - Bold section headers
        - Bullet points for lists
        - Tables for comparative data
        - Page breaks between sections

        **Quality Requirements:**
        - Minimum 4000 words
        - Every opposing claim addressed
        - All citations verifiable
        - Neutral tone
        """
    )
    await asyncio.sleep(0)

    progress(0.3, desc="Getting report")
    response = ""
    # for chunk in init_client(prompt, model_pro='groq', model_type='llama-3.3-70b-versatile', max_tokens=8000, stream=True):
    for chunk in init_client(prompt, model_pro='gpt', max_tokens=8000, stream=True):
        if generation_cancelled.is_set():
            progress(1.0, desc="Processing file")
            logging.info(f"Generation cancelled during rebuttal generation for user {username}")
            yield "Generation cancelled."
            return
        response += chunk.choices[0].delta.content or ""
        yield response
        await asyncio.sleep(0.01)
    logging.info(f"Generated combined rebuttal for user {username}")

    progress(0.8, desc="Getting report")
    user_data.memory.save_context({"input": input_text}, {"output": user_data.encrypt_data(response)})
    user_data.rebuttal_data = user_data.encrypt_data(response)
    progress(1.0, desc="Getting report")
    await asyncio.sleep(0)
    yield response