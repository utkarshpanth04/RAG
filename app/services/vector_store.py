# app/services/indexer.py
import os
import sys
import hashlib
import shutil
from pathlib import Path

# Dynamically add the project root directory to the Python path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from langchain_cohere import CohereEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
from app.core.config import (
    load_environment,
    DB_DIR,
    PDF_DIR,
    EMBEDDING_MODEL_NAME
)
from ingestion.loader import load_all_pdfs_from_dir

def _get_dir_hash(pdf_dir: str) -> str:
    """Returns a combined MD5 hash of all PDFs in the directory.
    
    Detects additions, removals, or modifications of any PDF.
    """
    hasher = hashlib.md5()
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))  # sorted for determinism
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")
    for pdf_path in pdf_files:
        # Include filename so renames/additions are detected
        hasher.update(pdf_path.name.encode())
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def create_or_load_vector_db():
    """Embeds the document chunks and saves them to a local ChromaDB.
    
    Automatically detects when the source PDF has changed (via MD5 hash)
    and rebuilds the database — no manual deletion needed.
    """
    # 1. Load the environment keys
    load_environment()

    # 2. Initialize Cohere Embeddings with the centralized model name
    print(f"[Indexer] Initializing Cohere Embeddings ({EMBEDDING_MODEL_NAME})...")
    embeddings = CohereEmbeddings(model=EMBEDDING_MODEL_NAME)

    HASH_FILE = os.path.join(DB_DIR, ".pdf_hash")
    current_hash = _get_dir_hash(PDF_DIR)

    # Check if DB exists AND was built from the same set of PDFs
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        if os.path.exists(HASH_FILE):
            with open(HASH_FILE, "r") as f:
                stored_hash = f.read().strip()
            if stored_hash == current_hash:
                print(f"[Indexer] PDFs unchanged. Loading existing ChromaDB from {DB_DIR}...")
                return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            else:
                print("[Indexer] PDF set has changed! Deleting old ChromaDB and rebuilding...")
                shutil.rmtree(DB_DIR)
        else:
            print("[Indexer] No hash record found. Rebuilding ChromaDB to ensure consistency...")
            shutil.rmtree(DB_DIR)

    # Load and chunk all PDFs from the data directory
    print("[Indexer] No existing DB found. Starting ingestion pipeline...")
    chunks = load_all_pdfs_from_dir(PDF_DIR)

    # 4. Create and persist Chroma database
    print(f"[Indexer] Embedding {len(chunks)} chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # 5. Save the PDF hash so future startups can detect changes
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)
    
    print(f"[Indexer] Database successfully created and saved to {DB_DIR}")
    return vectorstore

if __name__ == "__main__":
    # TEST BLOCK: Run this to build your database!
    try:
        db = create_or_load_vector_db()
    except Exception as e:
        print(f"[Indexer] Indexer building failed: {e}")
