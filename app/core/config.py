# app/core/config.py
import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# 1. Centralized Project Directories (using absolute paths)
# Config is located at app/core/config.py. To reach root, we go up three levels.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
DB_DIR = str(BASE_DIR / "chroma_db")
PDF_DIR = str(BASE_DIR / "data")  # All PDFs in this folder will be ingested

# 2. Centralized Model Configurations
EMBEDDING_MODEL_NAME = "embed-english-v3.0"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

def load_environment():
    """Loads the .env file using its absolute path and maps custom keys."""
    # Always load from the root .env path
    load_dotenv(dotenv_path=ENV_PATH) 

    # Map the custom Cohere key
    if "EMBEDDINGGS_KEY" in os.environ:
        os.environ["COHERE_API_KEY"] = os.environ["EMBEDDINGGS_KEY"]

    # Map the custom Groq key
    if "LLM_KEY" in os.environ:
        os.environ["GROQ_API_KEY"] = os.environ["LLM_KEY"]

    # Validation: Ensure keys are present in active memory
    if not os.environ.get("COHERE_API_KEY"):
        raise ValueError("Missing EMBEDDINGGS_KEY in .env file.")
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Missing LLM_KEY in .env file.")

if __name__ == "__main__":
    # TEST BLOCK: Verify variables and keys are resolved
    try:
        load_environment()
        print("✅ Environment loaded successfully!")
        print(f"Base Directory: {BASE_DIR}")
        print(f"Vector DB Path: {DB_DIR}")
        print(f"PDF Source Path: {PDF_PATH}")
        print(f"Cohere Key Mapped: {os.environ['COHERE_API_KEY'][:6]}...")
        print(f"Groq Key Mapped: {os.environ['GROQ_API_KEY'][:6]}...")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
