# app/services/indexer.py
import os
import sys
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
    PDF_PATH,
    EMBEDDING_MODEL_NAME
)
from ingestion.loader import load_and_chunk_pdf

def create_or_load_vector_db():
    """Embeds the document chunks and saves them to a local ChromaDB."""
    # 1. Load the environment keys
    load_environment()

    # 2. Initialize Cohere Embeddings with the centralized model name
    print(f"[Indexer] Initializing Cohere Embeddings ({EMBEDDING_MODEL_NAME})...")
    embeddings = CohereEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Check if DB already exists to avoid re-embedding (saves API costs)
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print(f"[Indexer] Existing ChromaDB found at {DB_DIR}. Loading it...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return vectorstore

    # 3. If no DB exists, load and chunk the PDF from the centralized path
    print("[Indexer] No existing DB found. Starting ingestion pipeline...")
    chunks = load_and_chunk_pdf(PDF_PATH)

    # 4. Create and persist Chroma database
    print(f"[Indexer] Embedding {len(chunks)} chunks and saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"[Indexer] Database successfully created and saved to {DB_DIR}")
    return vectorstore

if __name__ == "__main__":
    # TEST BLOCK: Run this to build your database!
    try:
        db = create_or_load_vector_db()
    except Exception as e:
        print(f"[Indexer] Indexer building failed: {e}")
