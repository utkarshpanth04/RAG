# app/schemas/query.py
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
