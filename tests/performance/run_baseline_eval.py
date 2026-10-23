"""
Baseline evaluation: same queries as run_pipeline_eval.py but sent
directly to qwen2.5-med via Ollama with no RAG, no routing, no enrichment.
Output: tests/performance/baseline_results.json
"""

import json
import time
import sys
import os
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import Config

RESULTS_PATH  = os.path.join(os.path.dirname(__file__), "results.json")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_results.json")

PROMPT_TEMPLATE = """You are a clinical language model specialised in tropical and infectious diseases.
Answer the following medical question accurately and concisely.
Clearly state when information is uncertain or when a clinician should be consulted.

Question: {query}

Answer:"""


def _disease_hit(answer: str, gt: dict) -> bool:
    a = answer.lower()
    targets = gt.get("diseases", []) + gt.get("key_terms", [])
    return any(t.lower() in a for t in targets if t)


def run_baseline():
    with open(RESULTS_PATH) as f:
        pipeline_results = json.load(f)

    baseline = []
    for i, r in enumerate(pipeline_results, 1):
        query = r["query"]
        gt_diseases = r.get("primary_disease", "")
        # Reconstruct a minimal ground truth from pipeline result
        gt = {"diseases": [gt_diseases], "key_terms": [gt_diseases]}

        print(f"[{i}/{len(pipeline_results)}] {query[:65]}…")
        start = time.time()
        try:
            resp = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": Config.FINETUNED_MODEL_NAME,
                    "prompt": PROMPT_TEMPLATE.format(query=query),
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 512},
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
            elapsed = round(time.time() - start, 2)
            hit = _disease_hit(answer, gt)
            baseline.append({
                "id": r["id"],
                "sample_id": r["sample_id"],
                "query_type": r["query_type"],
                "query": query,
                "primary_disease": gt_diseases,
                "answer_snippet": answer[:250],
                "disease_hit": hit,
                "latency_s": elapsed,
                "error": None,
            })
            print(f"  ✓ disease_hit={hit} [{elapsed}s]")
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            baseline.append({
                "id": r["id"],
                "sample_id": r["sample_id"],
                "query_type": r["query_type"],
                "query": query,
                "primary_disease": gt_diseases,
                "answer_snippet": "",
                "disease_hit": False,
                "latency_s": elapsed,
                "error": str(e),
            })
            print(f"  ✗ {e}")

    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    n = len(baseline)
    dis_hit = round(sum(1 for r in baseline if r["disease_hit"]) / n * 100, 1)
    avg_lat = round(sum(r["latency_s"] for r in baseline) / n, 2)
    print(f"\nBaseline — disease_hit={dis_hit}% avg_lat={avg_lat}s")
    print(f"Saved → {BASELINE_PATH}")


if __name__ == "__main__":
    run_baseline()
