# app/services/generator.py
import os
import sys
from pathlib import Path

# Dynamically add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import load_environment, LLM_MODEL_NAME, LLM_TEMPERATURE

def generate_answer(question: str, context: str) -> str:
    """Generates an answer using Groq based on the provided context."""
    # 1. Load keys
    load_environment()

    # 2. Define a strict System Prompt to prevent AI hallucinations
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert scientific researcher and academic assistant.\n"
            "Answer the user's question accurately, comprehensively, and objectively, "
            "relying ONLY on the provided research paper context below.\n"
            "If the answer cannot be found or inferred from the context, state clearly that "
            "the information is not available in the paper. Do not make up facts or equations.\n\n"
            "RESEARCH PAPER CONTEXT:\n{context}"
        )),
        ("human", "{question}")
    ])

    # 3. Initialize Groq LLM using centralized configurations
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,  # Low temperature keeps the model factual and grounded
    )

    # 4. Build and execute the chain
    print("[Generator] Asking Groq LLM...")
    chain = prompt_template | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content

async def generate_answer_stream(question: str, context: str):
    """Yields an answer chunk-by-chunk using Groq based on the provided context."""
    # 1. Load keys
    load_environment()

    # 2. Define a strict System Prompt to prevent AI hallucinations
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert scientific researcher and academic assistant.\n"
            "Answer the user's question accurately, comprehensively, and objectively, "
            "relying ONLY on the provided research paper context below.\n"
            "If the answer cannot be found or inferred from the context, state clearly that "
            "the information is not available in the paper. Do not make up facts or equations.\n\n"
            "RESEARCH PAPER CONTEXT:\n{context}"
        )),
        ("human", "{question}")
    ])

    # 3. Initialize Groq LLM using centralized configurations
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,  # Low temperature keeps the model factual and grounded
    )

    # 4. Build and execute the chain asynchronously
    print("[Generator] Asking Groq LLM (streaming)...")
    chain = prompt_template | llm
    async for chunk in chain.astream({"context": context, "question": question}):
        yield chunk.content

if __name__ == "__main__":
    # TEST BLOCK: Test generator with retrieved context
    import asyncio
    from app.services.retriever import get_relevant_context
    
    test_question = "What programming language is used for pest detection?"
    
    print("[Test] Running full pipeline test (sync)...")
    try:
        # Step 1: Retrieve context
        matched_docs = get_relevant_context(test_question, k=4)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])
        
        # Step 2: Generate response
        answer = generate_answer(test_question, context_text)
        print("\n[Generator] FINAL ANSWER FROM GROQ:")
        print("=" * 50)
        print(answer)
        print("=" * 50)
    except Exception as e:
        print(f"[Generator] Generator test failed: {e}")

    print("\n[Test] Running full pipeline test (async stream)...")
    async def run_stream():
        try:
            # Step 1: Retrieve context
            matched_docs = get_relevant_context(test_question, k=4)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])
            
            # Step 2: Generate response stream
            async for chunk in generate_answer_stream(test_question, context_text):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\n[Generator] Streaming test failed: {e}")

    asyncio.run(run_stream())
