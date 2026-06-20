# ingestion/loader.py
import os
import sys
from pathlib import Path

# Dynamically add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import PDF_PATH  # Import centralized PDF path

def load_and_chunk_pdf(file_path: str = PDF_PATH, chunk_size: int = 800, chunk_overlap: int = 150):
    """Loads a PDF and splits it into logical, overlapping chunks.
    
    Args:
        file_path (str): Path to the PDF file.
        chunk_size (int): Max character length of each chunk.
        chunk_overlap (int): Overlap size between adjacent chunks.
        
    Returns:
        list: A list of split Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find PDF at {file_path}")

    print(f"[Loader] Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    print("[Loader] Chunking document...")
    # Custom separators tailored to keep sections, paragraphs, and list bullets intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n•", "\n-", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(pages)
    return chunks

if __name__ == "__main__":
    # TEST BLOCK: Run this file directly to test the PDF parser
    try:
        # Uses the centralized absolute path from config.py
        document_chunks = load_and_chunk_pdf(PDF_PATH)
        print(f"[Loader] Successfully created {len(document_chunks)} chunks.")
        print("-" * 40)
        print("Preview of Chunk 1:\n", document_chunks[0].page_content)
    except Exception as e:
        print(f"[Loader] Loader verification failed: {e}")
