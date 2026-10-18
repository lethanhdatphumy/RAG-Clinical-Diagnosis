"""
Push filtered clinical case documents to a Qdrant Cloud collection.

Usage:
    uv run python -m src.embedding.qdrant_uploader

Environment variables required (in .env):
    QDRANT_HOST      — full URL of your Qdrant Cloud cluster
    QDRANT_API_KEY   — Qdrant Cloud API key
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config.settings import Config
from src.embedding.embedder import ClinicalEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class QdrantUploader:
    """Embeds filtered clinical cases and upserts them into a Qdrant collection."""

    def __init__(self):
        self.collection_name = Config.QDRANT_COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        self.batch_size = Config.EMBEDDING_BATCH_SIZE

        self.embedder = ClinicalEmbedder()
        self.model = SentenceTransformer(Config.TEXT_EMBEDDING_MODEL)
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )
        logger.info("Connected to Qdrant at %s", Config.QDRANT_URL)

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            logger.info("Collection '%s' already exists — skipping creation.", self.collection_name)
            return

        logger.info("Creating collection '%s'…", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )
        logger.info("Collection created.")

    def _build_payload(self, case: dict, page_content: str) -> dict:
        """Extract the fields to store as Qdrant point payload.

        page_content is the combined text used for embedding — required by
        LangChain's Qdrant vectorstore wrapper to reconstruct Documents.
        """
        return {
            "page_content": page_content,
            "case_id": case.get("case_id", ""),
            "diseases": case.get("diseases", []),
            "symptoms": case.get("symptoms", []),
            "treatments": case.get("treatments", []),
            "pathogens": case.get("pathogens", []),
            "risk_factors": case.get("risk_factors", []),
            "laboratory_findings": case.get("laboratory_findings", []),
            "vital_signs": case.get("vital_signs", []),
            "procedures": case.get("procedures", []),
            "patient_history": case.get("patient_history", ""),
        }

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the sentence transformer model."""
        logger.info("Embedding %d documents…", len(texts))
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
        ).tolist()

    def _build_points(
        self,
        cases: list[dict],
        texts: list[str],
        vectors: list[list[float]],
    ) -> list[PointStruct]:
        """Pair each case with its vector to create Qdrant PointStructs."""
        return [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, case["case_id"])),
                vector=vector,
                payload=self._build_payload(case, text),
            )
            for case, text, vector in zip(cases, texts, vectors)
        ]

    def _upsert_batches(self, points: list[PointStruct]) -> None:
        """Upsert points to Qdrant in batches."""
        total = len(points)
        for start in range(0, total, self.batch_size):
            batch = points[start : start + self.batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.info("Uploaded %d / %d points.", min(start + self.batch_size, total), total)

    def upload(self) -> None:
        """Full pipeline: load → embed → ensure collection → upsert → verify."""
        cases = self.embedder.load_filtered_cases()
        logger.info("Loaded %d filtered cases.", len(cases))

        texts = [self.embedder.prepare_text_for_embedding(case) for case in cases]
        vectors = self._embed(texts)

        self._ensure_collection()

        points = self._build_points(cases, texts, vectors)
        self._upsert_batches(points)

        info = self.client.get_collection(self.collection_name)
        logger.info(
            "Done — collection '%s' | status: %s | vectors: %s",
            self.collection_name,
            info.status,
            info.vectors_count,
        )

    def verify(self) -> None:
        """Print a quick summary of the collection state."""
        info = self.client.get_collection(self.collection_name)
        logger.info(
            "Collection '%s' | status: %s | vectors: %s",
            self.collection_name,
            info.status,
            info.vectors_count,
        )


if __name__ == "__main__":
    uploader = QdrantUploader()
    uploader.upload()
    uploader.verify()
