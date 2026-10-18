"""
Pipeline Evaluation Dataset Generator.

Generates a dataset for comparing plain LLM calls vs the full pipeline.
Each clinical case produces three query variants:
  - vague       : minimal symptoms, no demographics/travel (triggers Agentic path)
  - specific    : rich symptoms + demographics + travel (triggers Hybrid RAG path)
  - general     : knowledge question about the disease (triggers fine-tuned LLM path)

Each sample includes ground-truth diseases and key diagnostic terms so we can
score any response automatically.

Usage:
    python -m tests.evaluation.pipeline_eval_dataset            # generate + save
    python -m tests.evaluation.pipeline_eval_dataset --max-cases 10
    python -m tests.evaluation.pipeline_eval_dataset --show
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from config.settings import Config
from src.embedding.embedder import ClinicalEmbedder

_BULLET_RE = re.compile(r"^\s*[\*\-•]\s+")
_PREFIX_RE = re.compile(
    r"^(\*?Final (?:selection|choice|string|polish|answer|check|draft)[:\*]*\s*"
    r"|Query:\s*|Result:\s*|Only query:\s*|Let's go(?: with)?[.:]\s*"
    r"|-\s*Only query:\s*(?:Yes\.)?|This is the most realistic\."
    r"|Perfect[.!]?\s*|Done[.!]?\s*|Great[.!]?\s*|Sure[.!]?\s*"
    r"|Ok(?:ay)?[.!]?\s*|Alright[.!]?\s*|Here(?:\s+it\s+is)?[.:]\s*)",
    re.IGNORECASE,
)


def _clean(raw: str) -> str:
    """Extract the final answer from a verbose Gemma chain-of-thought response."""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    # Filter out bullet/star reasoning lines — keep plain prose lines only
    prose = [l for l in lines if not _BULLET_RE.match(l) and not l.startswith("*")]
    candidate_lines = prose if prose else lines

    # Take the last non-empty prose line (Gemma puts the final answer last)
    last = candidate_lines[-1] if candidate_lines else raw.strip()

    # Strip known prefixes and punctuation artifacts
    last = _PREFIX_RE.sub("", last).strip().strip('"').strip("*").strip('"').strip()

    # Strip short affirmation words glued to the front ("Perfect.Why...")
    last = re.sub(
        r"^(?:Perfect|Done|Great|Sure|Yes|Okay|Ok|Alright|Here(?:\s+it\s+is)?)[.!]?\s*",
        "", last, flags=re.IGNORECASE,
    ).strip()

    if last.startswith('"') and last.endswith('"'):
        last = last[1:-1].strip()

    # Exact duplicate: "SentenceSentence"
    n = len(last)
    if n > 0 and n % 2 == 0:
        mid = n // 2
        if last[:mid] == last[mid:]:
            return last[:mid].strip()

    # Near-duplicate: two similar sentences joined without newline
    mid_start, mid_end = n // 3, 2 * n // 3
    for i in range(mid_start, mid_end):
        if last[i] == ".":
            first_half = last[:i + 1].strip()
            second_half = last[i + 1:].strip()
            if second_half and first_half:
                shorter, longer = sorted([first_half, second_half], key=len)
                if longer and len(shorter) / len(longer) > 0.7:
                    return first_half

    return last


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("tests/evaluation/pipeline_eval_dataset.json")

QUERY_TYPES = ("vague", "specific", "general")


class PipelineEvalDatasetGenerator:
    """
    Builds a labelled evaluation dataset for end-to-end pipeline comparison.

    Each sample:
        sample_id        — unique identifier  e.g. "pipe_001_vague"
        source_case_id   — which filtered case this came from
        query_type       — "vague" | "specific" | "general"
        query            — the clinical question
        ground_truth     — {primary_disease, diseases, key_terms, diagnosis_category}
    """

    _METADATA_PREFIXES = ("foreword", "copyright", "index", "preface", "contents", "acknowledgement")

    def __init__(self):
        self.embedder = ClinicalEmbedder()
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.model = Config.GEMINI_MODEL

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_clinical_case(self, case: dict) -> bool:
        case_id = case.get("case_id", "").lower()
        has_symptoms = bool(case.get("symptoms"))
        has_history = bool(case.get("patient_history", "").strip())
        is_metadata = any(case_id.startswith(p) for p in self._METADATA_PREFIXES)
        return (has_symptoms or has_history) and not is_metadata

    def _primary_disease(self, case: dict) -> str:
        diseases = case.get("diseases", [])
        return diseases[0] if diseases else "unknown"

    def _diagnosis_category(self, diseases: list[str]) -> str:
        text = " ".join(diseases).lower()
        if any(w in text for w in ("hypertension", "heart failure", "cardiac", "anaemia", "sickle", "haemolytic", "renal failure")):
            return "non-infectious"
        if any(w in text for w in ("virus", "viral", "ebola", "dengue", "hiv", "hepatitis")):
            return "viral"
        if any(w in text for w in ("malaria", "plasmodium", "leishmania", "trypanosoma", "parasite", "schistosoma")):
            return "parasitic"
        if any(w in text for w in ("bacteria", "typhoid", "leptospira", "salmonella", "tuberculosis", "tb")):
            return "bacterial"
        if any(w in text for w in ("fungal", "aspergillus", "candida", "cryptococcus")):
            return "fungal"
        return "infectious"

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _vague_prompt(self, case: dict) -> str:
        symptoms = ", ".join(case.get("symptoms", [])[:3]) or "fever"
        return f"""Write ONE vague clinical query a non-expert patient might type.
Symptoms available: {symptoms}

Strict rules:
- Use only 1-2 symptoms, in plain everyday language
- No travel history, no age/gender, no disease names
- Maximum 15 words
- Output the query only — no reasoning, no explanation, no alternatives"""

    def _specific_prompt(self, case: dict) -> str:
        diseases = ", ".join(case.get("diseases", [])[:3]) or "unknown"
        symptoms = ", ".join(case.get("symptoms", [])[:8]) or "unknown"
        history = case.get("patient_history", "")[:300]
        risk_factors = ", ".join(case.get("risk_factors", [])[:4]) or "none"
        treatments = ", ".join(case.get("treatments", [])[:3]) or "none"

        return f"""Write ONE specific clinical search query a physician would use to find similar cases.

Case:
  History     : {history}
  Diseases    : {diseases}
  Symptoms    : {symptoms}
  Risk factors: {risk_factors}
  Treatments  : {treatments}

Rules:
- Include symptoms, demographics (age/gender if known), travel/exposure
- Do NOT name the disease directly
- 1-3 sentences, medical tone
- Output the query only — no reasoning, no explanation, no alternatives"""

    def _general_prompt(self, case: dict) -> str:
        disease = self._primary_disease(case)
        return f"""Write ONE general medical knowledge question about {disease}.

Rules:
- Cover mechanisms, treatment protocols, epidemiology, or diagnostic criteria
- No specific patient reference
- 1 sentence ending with a question mark
- Output the question only — no reasoning, no explanation, no alternatives"""

    def _key_terms_prompt(self, case: dict) -> str:
        diseases = ", ".join(case.get("diseases", [])[:5]) or "unknown"
        symptoms = ", ".join(case.get("symptoms", [])[:6]) or "unknown"
        treatments = ", ".join(case.get("treatments", [])[:4]) or "none"
        lab = ", ".join(case.get("laboratory_findings", [])[:4]) or "none"

        return f"""List the 5-8 most important diagnostic keywords for this clinical case.
Diseases: {diseases}
Symptoms: {symptoms}
Treatments: {treatments}
Lab: {lab}

Output a JSON array of lowercase strings only. Example: ["malaria", "blood smear", "artemisinin"]
No explanation, no markdown, no reasoning — just the JSON array."""

    # ── Generation ────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, case_id: str, label: str, max_retries: int = 3) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return _clean(response.text.strip())
            except Exception as exc:
                wait = 2 ** attempt
                if attempt < max_retries:
                    logger.warning("[%s/%s] attempt %d failed: %s — retrying in %ds", case_id, label, attempt, exc,
                                   wait)
                    time.sleep(wait)
                else:
                    logger.warning("[%s/%s] all retries failed: %s", case_id, label, exc)
        return ""

    def _generate_key_terms(self, case: dict) -> list[str]:
        raw = self._call_llm(self._key_terms_prompt(case), case["case_id"], "key_terms")
        # Find a JSON array anywhere in the response
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                terms = json.loads(match.group())
                if isinstance(terms, list):
                    return [str(t).lower().strip() for t in terms if str(t).strip()]
            except json.JSONDecodeError:
                pass
        # fallback: pull quoted strings
        quoted = re.findall(r'"([^"]+)"', raw)
        if quoted:
            return [t.lower().strip() for t in quoted[:8]]
        return []

    def generate_sample(self, case: dict, idx: int) -> list[dict]:
        """Generate all 3 query variants for one case."""
        case_id = case["case_id"]
        diseases = case.get("diseases", [])
        primary = self._primary_disease(case)
        category = self._diagnosis_category(diseases)

        key_terms = self._generate_key_terms(case)
        time.sleep(1)

        samples = []
        prompts = {
            "vague": self._vague_prompt(case),
            "specific": self._specific_prompt(case),
            "general": self._general_prompt(case),
        }

        for qtype, prompt in prompts.items():
            query = self._call_llm(prompt, case_id, qtype)
            if not query:
                symptoms = ", ".join(case.get("symptoms", [])[:2])
                history = case.get("patient_history", "")[:80].strip()
                if qtype == "vague":
                    query = f"Patient with {symptoms}" if symptoms else f"I feel unwell with {primary}"
                elif qtype == "specific":
                    query = history if history else f"Patient with {symptoms} and history of {primary}"
                else:
                    query = f"What is the standard treatment for {primary}?"

            samples.append({
                "sample_id": f"pipe_{idx:03d}_{qtype}",
                "source_case_id": case_id,
                "query_type": qtype,
                "query": query,
                "ground_truth": {
                    "primary_disease": primary,
                    "diseases": diseases[:8],
                    "key_terms": key_terms,
                    "diagnosis_category": category,
                },
            })
            time.sleep(2)

        return samples

    def generate(self, max_cases: int | None = None) -> list[dict]:
        cases = self.embedder.load_filtered_cases()
        logger.info("Loaded %d cases.", len(cases))

        clinical_cases = [c for c in cases if self._is_clinical_case(c)]
        logger.info("%d clinical cases after filtering.", len(clinical_cases))

        if max_cases:
            clinical_cases = clinical_cases[:max_cases]

        all_samples = []
        for i, case in enumerate(clinical_cases):
            logger.info("[%d/%d] Generating queries for: %s", i + 1, len(clinical_cases), case["case_id"])
            samples = self.generate_sample(case, i + 1)
            all_samples.extend(samples)

        logger.info("Generated %d total samples (%d cases × 3 query types).", len(all_samples), len(clinical_cases))
        return all_samples

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
    parser = argparse.ArgumentParser(description="Generate pipeline evaluation dataset")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases (default: all)")
    parser.add_argument("--show", action="store_true", help="Print 3 sample entries and exit")
    args = parser.parse_args()

    if args.show and OUTPUT_PATH.exists():
        samples = PipelineEvalDatasetGenerator.load()
        for s in samples[:3]:
            print(json.dumps(s, indent=2))
        return

    gen = PipelineEvalDatasetGenerator()
    samples = gen.generate(max_cases=args.max_cases)
    gen.save(samples)


if __name__ == "__main__":
    main()
