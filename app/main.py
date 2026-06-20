# app/main.py
import os
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from app.core.config import load_environment, DB_DIR
from app.services.vector_store import create_or_load_vector_db
from app.api.chat_routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load environment variables and initialize vector DB
    try:
        load_environment()
        print("[Startup] Initializing RAG Database...")
        create_or_load_vector_db()
        print("[Startup] RAG Database initialized successfully.")
    except Exception as e:
        print(f"[Startup] Configuration/Database Error during API startup: {e}")
        # We let the startup proceed so developers can inspect /health, 
        # or we could sys.exit(1) if desired.
    yield
    # Shutdown logic (if any) goes here

# 2. Initialize the FastAPI application
app = FastAPI(
    title="RAG-based Research Paper QA Engine API",
    description="An API that answers questions based on a localized vector database of research papers.",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Include API endpoints router with /api prefix
app.include_router(api_router, prefix="/api")

# 4. Health check endpoint to verify the server status
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0
    }
