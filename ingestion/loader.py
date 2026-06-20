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
from app.core.config import PDF_DIR  # Import centralized PDF directory

def load_and_chunk_pdf(file_path: str, chunk_size: int = 800, chunk_overlap: int = 150):
    """Loads a single PDF and splits it into logical, overlapping chunks.
    
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n•", "\n-", "\n", " ", ""]
    )
    return text_splitter.split_documents(pages)


def load_all_pdfs_from_dir(pdf_dir: str = PDF_DIR, chunk_size: int = 800, chunk_overlap: int = 150):
    """Loads ALL PDFs from a directory and returns combined chunks.
    
    Args:
        pdf_dir (str): Path to the folder containing PDF files.
        chunk_size (int): Max character length of each chunk.
        chunk_overlap (int): Overlap size between adjacent chunks.
        
    Returns:
        list: Combined list of split Document objects from all PDFs.
    """
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")

    print(f"[Loader] Found {len(pdf_files)} PDF(s) in {pdf_dir}:")
    for f in pdf_files:
        print(f"  - {f.name}")

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = load_and_chunk_pdf(str(pdf_path), chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    print(f"[Loader] Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    # TEST BLOCK: Run this file directly to test the PDF parser
    try:
        document_chunks = load_all_pdfs_from_dir(PDF_DIR)
        print(f"[Loader] Successfully created {len(document_chunks)} total chunks.")
        print("-" * 40)
        print("Preview of Chunk 1:\n", document_chunks[0].page_content)
    except Exception as e:
        print(f"[Loader] Loader verification failed: {e}")
