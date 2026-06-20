# app/api/chat_routes.py
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
from app.schemas.query import QueryRequest
from app.schemas.response import QueryResponse
from app.services.rag_service import query, query_stream

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_paper(request: QueryRequest):
    """
    Accepts a question in the request body, processes it through the 
    RAG pipeline, and returns the response.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The 'question' field cannot be empty.")
    
    try:
        # Route logic only calls the rag_service layer
        answer = query(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal RAG Engine Error: {str(e)}")

@router.post("/query/stream")
async def query_paper_stream(request: QueryRequest):
    """
    Accepts a question in the request body, processes it through the RAG pipeline,
    and returns a StreamingResponse yielding the answer chunk-by-chunk.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The 'question' field cannot be empty.")
    
    try:
        async def event_generator():
            async for chunk in query_stream(request.question):
                yield chunk
        return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal RAG Engine Error: {str(e)}")
