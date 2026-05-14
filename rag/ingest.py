"""RAG Ingestion Script — loads knowledge_base/ into ChromaDB.

Run this ONCE before starting the app:
    python -m rag.ingest
"""

import os
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# --- Configuration ---
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
CHROMA_DB_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "interview_knowledge"
CHUNK_SIZE = 500  # tokens (approximate via words * 1.3)
CHUNK_OVERLAP = 80


# --- Metadata mapping ---
# Maps folder names and file prefixes to metadata
FOLDER_TO_DOC_TYPE = {
    "question_patterns": "question_pattern",
    "rubrics": "rubric",
    "frameworks": "framework",
}

FILE_TO_CATEGORY = {
    "behavioral": "behavioral",
    "technical": "technical",
    "situational": "situational",
    "case_analysis": "case",
    "case": "case",
    "leadership": "leadership",
    "culture_fit": "culture_fit",
    "communication": "communication",
    "star_method": "behavioral",
    "competency_mapping": "general",
    "difficulty_calibration": "general",
}


def _detect_category(filename: str) -> str:
    """Detect interview category from filename."""
    name = filename.lower().replace(".md", "")
    for key, category in FILE_TO_CATEGORY.items():
        if key in name:
            return category
    return "general"


def _detect_difficulty(text: str) -> str:
    """Detect difficulty level from chunk text content."""
    text_lower = text.lower()
    if "### hard" in text_lower or "senior-level" in text_lower:
        return "hard"
    elif "### easy" in text_lower or "entry-level" in text_lower:
        return "easy"
    return "medium"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by approximate token count."""
    words = text.split()
    # Approximate: 1 token ≈ 0.75 words, so chunk_size tokens ≈ chunk_size * 0.75 words
    words_per_chunk = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)

    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += words_per_chunk - overlap_words

    return chunks


def ingest():
    """Load all knowledge base documents into ChromaDB."""
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[*] Initializing ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Delete existing collection if it exists (fresh ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_docs = []
    all_metadatas = []
    all_ids = []
    all_embeddings = []

    # Walk through knowledge_base directory
    for folder in KNOWLEDGE_BASE_DIR.iterdir():
        if not folder.is_dir():
            continue

        doc_type = FOLDER_TO_DOC_TYPE.get(folder.name, "unknown")

        for md_file in folder.glob("*.md"):
            print(f"   [+] Processing: {folder.name}/{md_file.name}")
            text = md_file.read_text(encoding="utf-8")
            category = _detect_category(md_file.name)
            chunks = _chunk_text(text)

            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{md_file.name}_{i}".encode()
                ).hexdigest()

                difficulty = _detect_difficulty(chunk)

                all_docs.append(chunk)
                all_metadatas.append({
                    "doc_type": doc_type,
                    "category": category,
                    "difficulty": difficulty,
                    "source_file": md_file.name,
                    "chunk_index": i,
                })
                all_ids.append(chunk_id)

    if not all_docs:
        print("[!] No documents found in knowledge_base/!")
        return

    print(f"\n[*] Embedding {len(all_docs)} chunks...")
    all_embeddings = model.encode(all_docs, show_progress_bar=True).tolist()

    print("[*] Storing in ChromaDB...")
    # ChromaDB has a batch limit, so we add in batches
    batch_size = 100
    for i in range(0, len(all_docs), batch_size):
        end = min(i + batch_size, len(all_docs))
        collection.add(
            documents=all_docs[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end],
            embeddings=all_embeddings[i:end],
        )

    print(f"\n[OK] Ingestion complete! {len(all_docs)} chunks stored in '{COLLECTION_NAME}'")
    print(f"   Categories: {set(m['category'] for m in all_metadatas)}")
    print(f"   Doc types: {set(m['doc_type'] for m in all_metadatas)}")


if __name__ == "__main__":
    ingest()
