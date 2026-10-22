from typing import Optional
from pydantic import BaseModel


class DiagnoseRequest(BaseModel):
    query: str
    user_context: Optional[str] = None


class SourceDocument(BaseModel):
    case_id: str
    diseases: list[str]
    symptoms: list[str]
    snippet: str


class DiagnoseResponse(BaseModel):
    answer: Optional[str]
    route: Optional[str]
    route_reasoning: Optional[str]
    is_specific: Optional[bool]
    enriched_query: Optional[str]
    assumptions: Optional[list[str]]
    docs_relevant: Optional[bool]
    source_documents: list[SourceDocument]
    model_used: Optional[str]
    path_taken: list[str]
    error: Optional[str]


class ChatRequest(BaseModel):
    message: str
    user_context: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    path_taken: list[str]
    model_used: Optional[str]
    source_cases: list[str]          # just case_ids for the UI to display
    enriched_query: Optional[str]
    assumptions: Optional[list[str]]
    error: Optional[str]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    finetuned_backend: str
