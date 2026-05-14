"""RAG Retriever — provides retrieval functions used by all agents."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --- Configuration ---
CHROMA_DB_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "interview_knowledge"

# Lazy-loaded singletons
_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    """Lazy-load the ChromaDB collection (singleton)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def get_frameworks(categories: list[str] | None = None, top_k: int = 3) -> str:
    """Retrieve interview framework documents.

    Used by the Session Planner to get universal interview structures
    (STAR method, competency mapping, difficulty calibration).

    Args:
        categories: Optional list of categories to filter by.
        top_k: Number of results to return.

    Returns:
        Formatted string of retrieved framework content.
    """
    collection = _get_collection()
    model = _get_model()

    query = "interview framework methodology structure competency assessment"
    query_embedding = model.encode([query]).tolist()

    where_filter = {"doc_type": "framework"}
    if categories:
        where_filter = {
            "$and": [
                {"doc_type": "framework"},
                {"category": {"$in": categories + ["general"]}},
            ]
        }

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
        )
    except Exception:
        # Fallback: query without category filter
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"doc_type": "framework"},
        )

    return _format_results(results)


def get_question_patterns(
    category: str, difficulty: str = "medium", top_k: int = 3
) -> str:
    """Retrieve question pattern templates by category.

    Used by the Interviewer to get structural inspiration for questions.
    The LLM adapts these universal patterns to the specific role.

    Args:
        category: Interview category (behavioral, technical, situational, etc.)
        difficulty: Difficulty level (easy, medium, hard)
        top_k: Number of results to return.

    Returns:
        Formatted string of retrieved question patterns.
    """
    collection = _get_collection()
    model = _get_model()

    query = f"{category} interview questions {difficulty} level"
    query_embedding = model.encode([query]).tolist()

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={
                "$and": [
                    {"doc_type": "question_pattern"},
                    {"category": category},
                ]
            },
        )
    except Exception:
        # Fallback: just filter by doc_type
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"doc_type": "question_pattern"},
        )

    return _format_results(results)


def get_rubric(category: str, top_k: int = 3) -> str:
    """Retrieve scoring rubric for a given interview category.

    Used by the Evaluator to ground scoring in documented criteria.

    Args:
        category: Interview category (behavioral, technical, case, etc.)
        top_k: Number of results to return.

    Returns:
        Formatted string of retrieved rubric content.
    """
    collection = _get_collection()
    model = _get_model()

    query = f"{category} scoring rubric evaluation criteria"
    query_embedding = model.encode([query]).tolist()

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={
                "$and": [
                    {"doc_type": "rubric"},
                    {"category": {"$in": [category, "communication", "general"]}},
                ]
            },
        )
    except Exception:
        # Fallback
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"doc_type": "rubric"},
        )

    return _format_results(results)


def _format_results(results: dict) -> str:
    """Format ChromaDB query results into a readable string for LLM context."""
    if not results or not results.get("documents") or not results["documents"][0]:
        return "No relevant context found in knowledge base."

    docs = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]

    formatted_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        source = meta.get("source_file", "unknown")
        category = meta.get("category", "unknown")
        formatted_parts.append(
            f"--- Context {i+1} [{category} | {source}] ---\n{doc}"
        )

    return "\n\n".join(formatted_parts)
