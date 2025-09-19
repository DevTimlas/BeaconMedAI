import sys
import os
from typing import Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import logging

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

current_vectorstore = None


def initialize_vectorstore(
        edition: str,
        persist_directory: str = None,
        force_recreate: bool = False,
        chunk_size: int = 8000,
        chunk_overlap: int = 200,
        chapter_ranges: Dict[str, tuple] = None
) -> Optional[Chroma]:
    try:
        # Resolve paths relative to the script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        edition_configs = {
            "6th Edition": {
                "persist_directory": os.path.join(base_dir, "data", "vectorstore_8k_batch_6th"),
                "chapter_ranges": {
                    "Chapter 1": (26, 43),
                    "Chapter 2": (44, 55),
                    "Chapter 3": (56, 71),
                    "Chapter 4": (72, 101),
                    "Chapter 5": (102, 125),
                    "Chapter 6": (126, 153),
                    "Chapter 7": (154, 183),
                    "Chapter 8": (184, 207),
                    "Chapter 9": (208, 237),
                    "Chapter 10": (238, 271),
                    "Chapter 11": (272, 304),
                    "Chapter 12": (305, 344),
                    "Chapter 13": (345, 371),
                    "Chapter 14": (373, 407),
                    "Chapter 15": (408, 517),
                    "Chapter 16": (518, 581),
                    "Chapter 17": (582, 627),
                }
            },
            "4th Edition": {
                "persist_directory": os.path.join(base_dir, "data", "vectorstore_8k_batch_4th"),
                "chapter_ranges": {
                    "Chapter 1": (30, 38),
                    "Chapter 2": (39, 46),
                    "Chapter 3": (47, 275),
                    "Chapter 4": (276, 302),
                    "Chapter 5": (303, 328),
                    "Chapter 6": (329, 380),
                    "Chapter 7": (381, 390),
                    "Chapter 8": (391, 414),
                    "Chapter 9": (415, 433),
                    "Chapter 10": (434, 456),
                    "Chapter 11": (457, 478),
                    "Chapter 12": (479, 499),
                    "Chapter 13": (500, 520),
                    "Chapter 14": (521, 538),
                    "Chapter 15": (539, 556),
                }
            }
        }

        if edition not in edition_configs:
            raise ValueError(f"Invalid edition: {edition}. Must be '4th Edition' or '6th Edition'.")

        config = edition_configs[edition]
        persist_directory = persist_directory or config["persist_directory"]

        # Load existing vectorstore
        if os.path.exists(persist_directory):
            logging.info(f"Loading existing vectorstore from {persist_directory}")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings(openai_api_key=api_key)
            )

        # If vectorstore doesn't exist, log error and return None
        logging.error(f"Vectorstore directory not found: {persist_directory}. Cannot initialize vectorstore.")
        raise FileNotFoundError(f"Vectorstore directory not found: {persist_directory}")

    except Exception as e:
        logging.error(f"Failed to initialize vectorstore for {edition}: {e}")
        return None


def initialize_vectorstore_for_edition(edition, force_recreate=False):
    global current_vectorstore
    logging.info(f"Initializing vectorstore for {edition}")
    current_vectorstore = initialize_vectorstore(
        edition=edition,
        force_recreate=force_recreate,
        chunk_size=8000,
        chunk_overlap=200
    )
    if current_vectorstore is None:
        logging.error(f"Failed to initialize vectorstore for {edition}")
        raise RuntimeError(f"Could not initialize vectorstore for {edition}")
    return current_vectorstore


def rerank_documents(query, documents, top_k=5):
    embedder = OpenAIEmbeddings(openai_api_key=api_key)
    query_embedding = embedder.embed_query(query)
    doc_embeddings = embedder.embed_documents([doc.page_content for doc in documents])

    similarities = [
        np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]

    doc_score_pairs = list(zip(documents, similarities))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in doc_score_pairs[:top_k]]


def get_ensemble_retriever(query, k=5, fetch_k=25, lambda_mult=0.5, metadata_filter=None):
    search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    if metadata_filter and "chapter" in metadata_filter:
        search_kwargs["filter"] = {"chapter": {"$in": metadata_filter["chapter"]}}
    vector_retriever = current_vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    collection_data = current_vectorstore._collection.get(include=["documents"])
    docs = collection_data.get("documents", [])
    bm25_retriever = BM25Retriever.from_texts(docs, k=k)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
