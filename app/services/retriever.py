# app/services/retriever.py
import os
import sys
from pathlib import Path

# Dynamically add the project root directory to the Python path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from langchain_cohere import CohereEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
from app.core.config import load_environment, DB_DIR, EMBEDDING_MODEL_NAME

def get_relevant_context(query: str, k: int = 4):
    """Searches ChromaDB and returns the top k relevant text chunks."""
    # Ensure environment keys are loaded
    load_environment()

    # Initialize embedding model using the centralized model name
    embeddings = CohereEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Load the persistent database from disk using the centralized path
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(f"ChromaDB directory not found at {DB_DIR}. Please run indexer.py first.")
        
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Perform similarity search
    print(f"[Retriever] Searching database for: '{query}'")
    docs = vectorstore.similarity_search(query, k=k)
    
    return docs

if __name__ == "__main__":
    # TEST BLOCK: Test if retrieval is finding relevant text
    test_query = "What programming language is used for pest detection?"
    try:
        results = get_relevant_context(test_query, k=2)
        print(f"[Retriever] Found {len(results)} matching chunks.")
        print("-" * 40)
        for i, doc in enumerate(results):
            print(f"[Chunk {i+1}] (Page {doc.metadata.get('page', 'Unknown')}):\n{doc.page_content}\n")
    except Exception as e:
        print(f"[Retriever] Retrieval test failed: {e}")
