import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Google Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Use the latest model
    GEMINI_MODEL = "models/gemma-4-26b-a4b-it"  # Fast and free

    # GEMINI_MODEL = "gemini-2.5-pro"  # More powerful but has rate limits

    # Embedding Models
    TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Paths
    RAW_DATA_DIR = "data/raw/case_reports"
    PROCESSED_DATA_DIR = "data/processed"
    VECTOR_STORE_DIR = "data/vector_store"
    # Qdrant cloud service
    QDRANT_URL = os.getenv("QDRANT_HOST", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "clinical_cases")

    # Embedding settings
    # 384 is fixed by all-MiniLM-L6-v2 — update if you swap the model
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    # Retrieval Settings
    TOP_K_RETRIEVAL = 3  # Number of top similar cases to retrieve

    # Fine-tuned model (served locally via Ollama)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    FINETUNED_MODEL_NAME = os.getenv("FINETUNED_MODEL_NAME", "gemma2-clinical:q4_k_m")
    OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))