"""
Fine-tuned LLM Node.

Tries endpoints in order:
  1. HuggingFace ZeroGPU Space  — if HF_INFERENCE_URL is set in .env
  2. Local Ollama                — fallback for local development

HF Space contract (POST {HF_INFERENCE_URL}/generate):
  Request : {"prompt": str, "max_tokens": int, "temperature": float}
  Response: {"generated_text": str}
"""

import logging
import requests
from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are a fine-tuned clinical language model specialised in
tropical and infectious diseases.

Answer the following medical question accurately and concisely.
Clearly state when information is uncertain or when a clinician should be consulted.

Question: {query}

Answer:"""


def _call_hf_space(prompt: str) -> str:
    """Call the HuggingFace ZeroGPU Space endpoint."""
    url = f"{Config.HF_INFERENCE_URL.rstrip('/')}/generate"
    response = requests.post(
        url,
        json={"prompt": prompt, "max_tokens": 512, "temperature": 0.7},
        timeout=Config.HF_INFERENCE_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("generated_text", "").strip()


def _call_ollama(prompt: str) -> str:
    """Call the local Ollama endpoint."""
    response = requests.post(
        f"{Config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": Config.FINETUNED_MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 512},
        },
        timeout=Config.OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def finetuned_node(state: AgentState) -> AgentState:
    """
    Calls the fine-tuned clinical LLM.
    Prefers HF ZeroGPU Space when HF_INFERENCE_URL is configured,
    falls back to local Ollama otherwise.
    """
    query = state["query"]
    path = list(state.get("path_taken") or []) + ["finetuned"]
    logger.info("Fine-tuned node — query: %s", query[:80])

    prompt = _PROMPT_TEMPLATE.format(query=query)
    use_hf = bool(Config.HF_INFERENCE_URL)

    try:
        if use_hf:
            logger.info("Fine-tuned node — using HF ZeroGPU Space: %s", Config.HF_INFERENCE_URL)
            answer = _call_hf_space(prompt)
            model_used = f"HF:{Config.HF_INFERENCE_URL}"
        else:
            logger.info("Fine-tuned node — using local Ollama: %s", Config.OLLAMA_BASE_URL)
            answer = _call_ollama(prompt)
            model_used = Config.FINETUNED_MODEL_NAME

        return {
            **state,
            "answer": answer,
            "source_documents": [],
            "model_used": model_used,
            "path_taken": path,
        }

    except requests.exceptions.ConnectionError as exc:
        if use_hf:
            msg = f"HF Space unreachable at {Config.HF_INFERENCE_URL}: {exc}"
        else:
            msg = (
                f"Fine-tuned model unreachable at {Config.OLLAMA_BASE_URL}. "
                "Ensure Ollama is running: `ollama serve`."
            )
        logger.error(msg)
        return {**state, "answer": None, "error": msg,
                "model_used": Config.HF_INFERENCE_URL or Config.FINETUNED_MODEL_NAME,
                "path_taken": path}

    except Exception as exc:
        logger.error("Fine-tuned node error: %s", exc)
        return {**state, "answer": None, "error": str(exc),
                "model_used": Config.HF_INFERENCE_URL or Config.FINETUNED_MODEL_NAME,
                "path_taken": path}
