"""
rag/ — Retrieval-Augmented Generation pipeline.

This package handles converting the knowledge_base/ markdown files
into searchable vector embeddings stored in ChromaDB.

HOW RAG WORKS IN THIS PROJECT:
==============================

1. INGESTION (run once via: python -m rag.ingest)
   - Reads 14 markdown files from knowledge_base/
   - Splits each file into ~300-400 token chunks (31 chunks total)
   - Converts each chunk into a 384-dimensional vector using sentence-transformers
   - Stores vectors + text + metadata in ChromaDB (local database in chroma_db/)

2. RETRIEVAL (happens automatically during interviews)
   - When an agent needs context, it calls a retriever function
   - The function converts the query into a vector
   - ChromaDB finds the most similar chunks using cosine similarity
   - Chunks are filtered by metadata (category, doc_type)
   - Top 3 most relevant chunks are returned as text

FILES:
- ingest.py    → One-time script to populate the vector database
- retriever.py → Functions that agents call to get relevant context
"""

from rag.retriever import get_frameworks, get_question_patterns, get_rubric

__all__ = ["get_frameworks", "get_question_patterns", "get_rubric"]
