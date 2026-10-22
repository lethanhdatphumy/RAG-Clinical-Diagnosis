"""
Clinical Diagnosis API

Endpoints:
  POST /diagnose  — full pipeline, returns all state fields
  POST /chat      — chat-friendly wrapper, returns reply + metadata
  GET  /health    — liveness + dependency check
"""

import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from api.schemas import (
    ChatRequest,
    ChatResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    SourceDocument,
)
from config.settings import Config
from src.agents.graph import run_diagnosis

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Clinical Diagnosis API starting up…")
    yield
    logger.info("Clinical Diagnosis API shutting down.")


app = FastAPI(
    title="Clinical Diagnosis API",
    description="Adaptive + Hybrid + Agentic RAG pipeline for clinical case diagnosis.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten when deploying to production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_source_docs(raw_docs: list | None) -> list[SourceDocument]:
    if not raw_docs:
        return []
    out = []
    for doc in raw_docs:
        out.append(SourceDocument(
            case_id=doc.get("case_id", "unknown"),
            diseases=doc.get("diseases", []),
            symptoms=doc.get("symptoms", []),
            snippet=doc.get("page_content", "")[:300],
        ))
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — also pings Qdrant and the fine-tuned LLM backend."""
    qdrant_status = "ok"
    try:
        client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        client.get_collections()
    except Exception as exc:
        qdrant_status = f"error: {exc}"

    finetuned_status = "ok"
    try:
        if Config.HF_INFERENCE_URL:
            url = f"{Config.HF_INFERENCE_URL.rstrip('/')}/gradio_api/info"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            finetuned_status = f"hf-space: {Config.HF_INFERENCE_URL}"
        else:
            resp = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=3)
            resp.raise_for_status()
            finetuned_status = f"ollama: {Config.OLLAMA_BASE_URL}"
    except Exception as exc:
        finetuned_status = f"unreachable: {exc}"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        finetuned_backend=finetuned_status,
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(request: DiagnoseRequest):
    """
    Run the full Adaptive + Hybrid + Agentic RAG pipeline.
    Returns all intermediate state fields for debugging and evaluation.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        state = run_diagnosis(
            query=request.query,
            user_context=request.user_context,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return DiagnoseResponse(
        answer=state.get("answer"),
        route=state.get("route"),
        route_reasoning=state.get("route_reasoning"),
        is_specific=state.get("is_specific"),
        enriched_query=state.get("enriched_query"),
        assumptions=state.get("assumptions"),
        docs_relevant=state.get("docs_relevant"),
        source_documents=_extract_source_docs(state.get("source_documents")),
        model_used=state.get("model_used"),
        path_taken=state.get("path_taken") or [],
        error=state.get("error"),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat-friendly endpoint for the UI.
    Accepts a message + optional user context, returns a reply and light metadata.
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    try:
        state = run_diagnosis(
            query=request.message,
            user_context=request.user_context,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    source_cases = [
        doc.get("case_id", "unknown")
        for doc in (state.get("source_documents") or [])
    ]

    return ChatResponse(
        reply=state.get("answer") or "I could not generate a response. Please try again.",
        path_taken=state.get("path_taken") or [],
        model_used=state.get("model_used"),
        source_cases=source_cases,
        enriched_query=state.get("enriched_query"),
        assumptions=state.get("assumptions"),
        error=state.get("error"),
    )
