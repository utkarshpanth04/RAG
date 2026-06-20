# app/schemas/response.py
from pydantic import BaseModel

class QueryResponse(BaseModel):
    question: str
    answer: str
