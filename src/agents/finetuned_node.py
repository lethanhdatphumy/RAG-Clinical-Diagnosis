import logging
import requests
from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_FINETUNED_PROMPT_TEMPLATE = """You are a fine-tuned clinical language model specialised in
tropical and infectious diseases.

Answer the following medical question accurately and concisely.
Clearly state when information is uncertain or when a clinician should be consulted.

Question: {query}

Answer:"""


def finetuned_node(state: AgentState) -> AgentState:
    """
    Calls the locally-running fine-tuned Gemma 2 9B model via the Ollama REST API.
    Falls back gracefully if the model is not available.
    """
    query = state["query"]
    logger.info("Fine-tuned node — query: %s", query[:80])

    prompt = _FINETUNED_PROMPT_TEMPLATE.format(query=query)

    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.FINETUNED_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512,
                },
            },
            timeout=Config.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        return {
            **state,
            "answer": answer,
            "source_documents": [],
            "model_used": Config.FINETUNED_MODEL_NAME,
        }

    except requests.exceptions.ConnectionError:
        msg = (
            f"Fine-tuned model unreachable at {Config.OLLAMA_BASE_URL}. "
            "Ensure Ollama is running: `ollama serve`."
        )
        logger.error(msg)
        return {**state, "answer": None, "error": msg, "model_used": Config.FINETUNED_MODEL_NAME}

    except Exception as exc:
        logger.error("Fine-tuned node error: %s", exc)
        return {**state, "answer": None, "error": str(exc), "model_used": Config.FINETUNED_MODEL_NAME}
