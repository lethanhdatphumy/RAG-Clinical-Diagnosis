"""
Synthetic Evaluation Dataset Generator for Chunked Context Evaluation.

Generates query → relevant_cases → irrelevant_cases triples from the
filtered clinical JSON cases. Each entry lets us measure:

  - Precision@K  : how many of the top-K retrieved chunks are relevant
  - Recall@K     : how many relevant chunks appear in the top-K
  - MRR          : mean reciprocal rank of the first relevant chunk
  - Hit@K        : whether at least one relevant chunk appears in top-K

Usage:
    python -m tests.evaluation.synthetic_dataset          # generate + save
    python -m tests.evaluation.synthetic_dataset --show   # print sample
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import google.generativeai as genai

from config.settings import Config
from src.embedding.embedder import ClinicalEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("tests/evaluation/synthetic_eval_dataset.json")
SEED = 42


class SyntheticDatasetGenerator:
    """
    Builds a labelled evaluation dataset for chunked-context retrieval.

    Each sample has:
        query            — a clinical question generated from a real case
        relevant_case_ids — case IDs whose chunks SHOULD be retrieved
        irrelevant_case_ids — case IDs that are NOT relevant (negative set)
        disease          — the primary disease in the source case
        source_case_id   — which case the query was generated from
    """

    def __init__(self, n_irrelevant: int = 4):
        self.n_irrelevant = n_irrelevant
        self.embedder = ClinicalEmbedder()

        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel(Config.GEMINI_MODEL)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_cases(self) -> list[dict]:
        cases = self.embedder.load_filtered_cases()
        logger.info("Loaded %d filtered cases.", len(cases))
        return cases

    # ── Query generation ─────────────────────────────────────────────────────

    def _build_query_prompt(self, case: dict) -> str:
        diseases = ", ".join(case.get("diseases", [])[:5]) or "unknown"
        symptoms = ", ".join(case.get("symptoms", [])[:8]) or "unknown"
        history = case.get("patient_history", "")[:300]
        risk_factors = ", ".join(case.get("risk_factors", [])[:4]) or "none"

        return f"""You are a clinical evaluation specialist.

Given this medical case summary, write ONE realistic clinical query that a
physician might type when looking for similar cases. The query should:
- Describe the patient in 1-2 sentences
- Mention key symptoms and travel/exposure history
- NOT name the disease directly
- Sound like a real diagnostic search query

Case details:
  Patient history : {history}
  Primary diseases: {diseases}
  Symptoms        : {symptoms}
  Risk factors    : {risk_factors}

Return ONLY the query string. No quotes, no explanation."""

    def generate_query_for_case(self, case: dict, max_retries: int = 3) -> str:
        prompt = self._build_query_prompt(case)
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.generate_content(prompt)
                # Handle both simple text and multi-part responses
                if response.parts:
                    return "".join(part.text for part in response.parts if hasattr(part, "text")).strip()
                return response.text.strip()
            except Exception as exc:
                wait = 2 ** attempt  # 2s, 4s, 8s
                if attempt < max_retries:
                    logger.warning(
                        "Attempt %d/%d failed for %s (%s) — retrying in %ds…",
                        attempt, max_retries, case.get("case_id"), exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "All retries failed for %s: %s — using fallback query.",
                        case.get("case_id"), exc,
                    )
        return self._fallback_query(case)

    # ── Relevance labelling ──────────────────────────────────────────────────

    def _find_relevant_cases(self, source_case: dict, all_cases: list[dict]) -> list[str]:
        """
        A case is relevant if it shares at least one primary disease with
        the source case (case-insensitive substring match).
        """
        source_diseases = {d.lower() for d in source_case.get("diseases", [])}
        relevant_ids = []

        for case in all_cases:
            if case["case_id"] == source_case["case_id"]:
                continue
            case_diseases = {d.lower() for d in case.get("diseases", [])}
            if source_diseases & case_diseases:
                relevant_ids.append(case["case_id"])

        # Always include the source case itself as relevant
        relevant_ids.insert(0, source_case["case_id"])
        return relevant_ids

    def _sample_irrelevant_cases(
        self,
        all_cases: list[dict],
        relevant_ids: list[str],
        rng: random.Random,
    ) -> list[str]:
        pool = [c["case_id"] for c in all_cases if c["case_id"] not in relevant_ids]
        return rng.sample(pool, min(self.n_irrelevant, len(pool)))

    # ── Dataset assembly ─────────────────────────────────────────────────────

    # ── Non-clinical case detection ───────────────────────────────────────────

    _METADATA_PREFIXES = ("foreword", "copyright", "index", "preface", "contents", "acknowledgement")

    def _is_clinical_case(self, case: dict) -> bool:
        """Return False for book metadata pages (Foreword, Copyright, etc.)."""
        case_id = case.get("case_id", "").lower()
        has_symptoms = bool(case.get("symptoms"))
        has_history = bool(case.get("patient_history", "").strip())
        is_metadata = any(case_id.startswith(p) for p in self._METADATA_PREFIXES)
        return (has_symptoms or has_history) and not is_metadata

    def _fallback_query(self, case: dict) -> str:
        """Build a simple query from symptoms when Gemini is skipped."""
        symptoms = ", ".join(case.get("symptoms", [])[:4])
        diseases = ", ".join(case.get("diseases", [])[:2])
        history = case.get("patient_history", "")[:100].strip()
        if history:
            return history
        if symptoms:
            return f"Patient with {symptoms}"
        return f"Case related to {diseases}" if diseases else "Unspecified clinical case"

    # ── Dataset assembly ──────────────────────────────────────────────────────

    def generate(self, max_cases: int | None = None) -> list[dict]:
        """
        Generate one evaluation sample for all 100 cases.
        Clinical cases get a Gemini-generated query; metadata pages use a fallback.
        """
        cases = self.load_cases()
        rng = random.Random(SEED)

        if max_cases:
            cases = cases[:max_cases]

        samples = []
        for i, case in enumerate(cases):
            is_clinical = self._is_clinical_case(case)
            logger.info(
                "[%d/%d] %s: %s",
                i + 1, len(cases),
                "Generating query" if is_clinical else "Fallback query (metadata page)",
                case["case_id"],
            )

            if is_clinical:
                query = self.generate_query_for_case(case)
                time.sleep(2)   # avoid Gemini rate limiting between requests
            else:
                query = self._fallback_query(case)
            relevant_ids = self._find_relevant_cases(case, cases)
            irrelevant_ids = self._sample_irrelevant_cases(cases, relevant_ids, rng)

            sample = {
                "sample_id": f"syn_{i+1:03d}",
                "source_case_id": case["case_id"],
                "query": query,
                "is_clinical": is_clinical,
                "primary_diseases": case.get("diseases", [])[:5],
                "relevant_case_ids": relevant_ids,
                "irrelevant_case_ids": irrelevant_ids,
            }
            samples.append(sample)

        logger.info("Generated %d evaluation samples.", len(samples))
        return samples

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, samples: list[dict], path: Path = OUTPUT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d samples to %s", len(samples), path)

    @staticmethod
    def load(path: Path = OUTPUT_PATH) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--show", action="store_true", help="Print 2 sample entries and exit")
    args = parser.parse_args()

    if args.show and OUTPUT_PATH.exists():
        samples = SyntheticDatasetGenerator.load()
        for s in samples[:2]:
            print(json.dumps(s, indent=2))
        return

    gen = SyntheticDatasetGenerator()
    samples = gen.generate(max_cases=args.max_cases)
    gen.save(samples)


if __name__ == "__main__":
    main()
