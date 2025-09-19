import sys
import os
import re
from typing import Optional, Dict, List, Set
from urllib.parse import urlparse, urlunparse
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import logging
from core.models import api_key, groq_client

load_dotenv()

openai_client = OpenAI()


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing trailing slashes, '/print', query parameters, and fragments.

    Args:
        url: The URL to normalize

    Returns:
        Normalized URL string
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    if path.endswith('/print'):
        path = path[:-6]
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))
    return normalized


def rm_duplicate_url(all_urls: List[str]) -> str:
    """
    Remove duplicate URLs after normalization.

    Args:
        all_urls: List of URLs

    Returns:
        Newline-separated string of unique URLs
    """
    seen = set()
    unique_urls = []
    for url in all_urls:
        norm = normalize_url(url)
        if norm not in seen:
            seen.add(norm)
            unique_urls.append(url)
    return "\n".join(unique_urls)


def web_search_url(content: str, clt: str) -> str:
    """
    Perform a web search for resources related to the provided content, excluding SSA and focusing on AMA impairment guidelines or other relevant content.

    Args:
        content: Input content to search for
        clt: Client type ('gpt' or 'groq')

    Returns:
        Newline-separated string of unique URLs
    """
    search_prompt = f"""
    Search the web for recent resources, cite studies/guidelines related or not related to AMA impairment evaluation guidelines and not related to SSA for this content: {content}
    Return a list of URLs only, with no additional text or explanations. Include a variety of useful content, not only AMA impairment evaluation guidelines.
    """
    try:
        if clt == 'gpt':
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini-search-preview-2025-03-11",
                web_search_options={
                    "user_location": {
                        "type": "approximate",
                        "approximate": {
                            "country": "US",
                            "city": "New York",
                            "region": "New York",
                        }
                    },
                },
                messages=[
                    {
                        "role": "user",
                        "content": search_prompt,
                    }
                ],
                max_tokens=1000
            )
            return rm_duplicate_url(re.findall(r'https?://\S+', completion.choices[0].message.content))
        elif clt == 'groq':
            completion = groq_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": search_prompt,
                    }
                ],
                max_tokens=1000
            )
            return rm_duplicate_url(re.findall(r'https?://\S+', completion.choices[0].message.content))
        else:
            logging.error(f"Invalid client type: {clt}")
            return ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Web search failed: {e}")
        return ""


def summarize_content(content: str, model_type: str = "gpt-4o-mini") -> str:
    """
    Summarize the provided content, focusing on medical or legal contexts.

    Args:
        content: Input text to summarize
        model_type: Model to use for summarization

    Returns:
        Summarized text or error message
    """
    prompt = f"""
    Summarize the content of this text: {content}. The text might be related to medical or legal contexts.
    Provide a concise and clear summary of what is visible, including:
    - Key objects or elements (e.g., doctor and patient's information like name, age, gender, history, notable medical details)
    - Context or setting
    - Notable details or observations
    - Maximum 3000 words
    - No repetition or duplicates
    - Map all notable named entities
    - Correct typos, punctuation, etc.
    - Include all medical words/terms without omission
    """
    try:
        response = openai_client.chat.completions.create(
            model=model_type,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=5000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error summarizing content: {e}")
        return f"Failed to summarize content: {str(e)}"


def no_summarize_content(content: str, model_type: str = "gpt-4o-mini") -> str:
    """
    Extract a numbered list of medical terms from the provided content.

    Args:
        content: Input text to process
        model_type: Model to use for extraction

    Returns:
        Numbered list of medical terms or error message
    """
    prompt = f"""
    Extract and return a numbered list of all medical terms in this text: {content} line by line.
    Include only medical terms related to conditions, procedures, medications, or anatomical references.
    Do not include general words, names, or numbers.
    No additional explanation, just the list of medical terms.
    """
    try:
        response = openai_client.chat.completions.create(
            model=model_type,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=5000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error extracting medical terms: {e}")
        return f"Failed to extract medical terms: {str(e)}"


def convert_to_proper_markdown(text: str) -> str:
    """
    Convert text to properly formatted markdown.

    Args:
        text: Input text to format

    Returns:
        Formatted markdown text
    """
    from datetime import datetime
    # Clean up headers
    text = re.sub(r'### (.*?)\n', r'## \1\n', text)
    # Format lists and bullets
    text = re.sub(r'- \*\*(.*?):\*\*', r'- **\1:**', text)
    text = re.sub(r'- \*\*(.*?)\*\*', r'  - **\1**', text)
    # Format tables and data points
    text = re.sub(r'- (\w+): ([\w/%]+)', r'- `\1:` \2', text)
    # Clean up signature section
    text = re.sub(r'---\n\n\*\*Signature:\*\*', '---\n\n**Signature:**', text)
    # Format dates consistently
    date_matches = re.findall(r'\d{2}-[A-Za-z]{3}-\d{4}', text)
    for date_str in date_matches:
        try:
            date_obj = datetime.strptime(date_str, '%d-%b-%Y')
            text = text.replace(date_str, date_obj.strftime('%B %d, %Y'))
        except ValueError:
            continue
    # Format percentages
    text = re.sub(r'(\d+)%', r'\1\%', text)
    # Clean up empty lines
    text = '\n'.join([line for line in text.split('\n') if line.strip() != ''])
    # Add proper spacing between sections
    sections = text.split('\n\n')
    formatted_sections = []
    for section in sections:
        if section.startswith('#'):
            formatted_sections.append('\n' + section + '\n')
        else:
            formatted_sections.append(section)
    return '\n'.join(formatted_sections)


def clean_term(
        term: str,
        min_length: int = 3,
        stop_words: Set[str] = {'the', 'and', 'for', 'with', 'this', 'that'}
) -> Optional[str]:
    """
    Clean extracted terms efficiently.

    Args:
        term: Term to clean
        min_length: Minimum length for valid terms
        stop_words: Set of stop words to exclude

    Returns:
        Cleaned term or None if invalid
    """
    term = term.strip().lower()
    if len(term) < min_length or term in stop_words or re.match(r'.*[0-9!@#$%^&*(){}[\]]+.*', term):
        return None
    return term
