# app/services/rag_service.py
import sys
from pathlib import Path

# Dynamically add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.services.retriever import get_relevant_context
from app.services.generator import generate_answer, generate_answer_stream

def query(question: str) -> str:
    """Orchestrates RAG pipeline: retrieves context and generates an answer."""
    # 1. Retrieve context from retriever
    matched_docs = get_relevant_context(question, k=4)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])

    # 2. Generate response from generator using context
    return generate_answer(question, context_text)

async def query_stream(question: str):
    """Orchestrates RAG pipeline: retrieves context and yields chunks sequentially."""
    # 1. Retrieve context from retriever
    matched_docs = get_relevant_context(question, k=4)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])

    # 2. Generate response stream from generator using context
    async for chunk in generate_answer_stream(question, context_text):
        yield chunk
