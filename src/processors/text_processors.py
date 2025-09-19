import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import Levenshtein
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import logging

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def check_text_similarity(text1, text2, method='cosine'):
    """
    Compare similarity between two texts using specified method.

    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('cosine', 'difflib', 'levenshtein')

    Returns:
        Similarity score (0 to 1)
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0
    if not text1 or not text2:
        return 0.0
    text1 = text1.lower()
    text2 = text2.lower()

    if method == 'cosine':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    elif method == 'difflib':
        matcher = difflib.SequenceMatcher(None, text1, text2)
        similarity = matcher.ratio()
    elif method == 'levenshtein':
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        similarity = 1 - (Levenshtein.distance(text1, text2) / max_len)
    else:
        raise ValueError("Method must be 'cosine', 'difflib', or 'levenshtein'")
    return round(similarity, 4)


def check_moderation(query):
    """
    Check content for moderation issues using OpenAI's moderation API.

    Args:
        query: Text to check

    Returns:
        Moderation result or False if not flagged
    """
    client = OpenAI(api_key=api_key)
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=query,
        )
        result = response.results[0]
        if result.flagged:
            flagged_categories = [
                category.replace("_", " ")
                for category, value in result.categories.__dict__.items()
                if value
            ]
            return f"Flagged for: {', '.join(flagged_categories)}" if flagged_categories else "Flagged, but no specific categories identified"
        return False
    except Exception as e:
        logging.error(f"Moderation check failed: {e}")
        return False