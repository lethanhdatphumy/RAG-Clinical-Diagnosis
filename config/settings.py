import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Google Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Use the latest model
    GEMINI_MODEL = "models/gemma-3-27b-it"  # Fast and free

    # GEMINI_MODEL = "gemini-2.5-pro"  # More powerful but has rate limits

    # Embedding Models
    TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Paths
    RAW_DATA_DIR = "data/raw/case_reports"
    PROCESSED_DATA_DIR = "data/processed"
    VECTOR_STORE_DIR = "data/vector_store"

    # Retrieval Settings
    TOP_K_RETRIEVAL = 3  # Number of top similar cases to retrieve