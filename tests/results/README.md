# Test Results Storage

This directory contains the JSON results from all baseline and RAG system tests.

## Files in This Directory

### Baseline Test Results

**File**: `baseline_gemma_20260601_121422.json` (40 KB)
- **Date**: June 1, 2026 @ 12:14:22 UTC
- **Model**: Gemma-4-26B (`models/gemma-4-26b-a4b-it`)
- **Type**: Baseline (Without RAG)
- **Test Cases**: 15
- **Success Rate**: 100% (15/15 completed)
- **Diagnosis Accuracy**: 53.33% (8/15 correct)
- **Keyword Match**: 83.33%
- **Avg Inference Time**: 11.41s

### RAG System Test Results

**File**: `rag_system_20260601_123213.json` (41 KB)
- **Date**: June 1, 2026 @ 12:32:13 UTC
- **Model**: Gemma-4-26B (`models/gemma-4-26b-a4b-it`)
- **Type**: RAG System (With Retrieval-Augmented Generation)
- **Test Cases**: 15
- **Success Rate**: 100% (15/15 completed)
- **Diagnosis Accuracy**: 93.33% (14/15 correct)
- **Keyword Match**: 76.67%
- **Precision@5**: 62.22%

### Additional Files

- `comparison_report_YYYYMMDD_HHMMSS.json` — Baseline vs RAG comparison reports

---

## Performance Summary

| Metric | Baseline (No RAG) | RAG System | Change |
|---|---|---|---|
| Diagnosis Accuracy | 53.33% (8/15) | **93.33% (14/15)** | **+40 pp** |
| Keyword Match | 83.33% | 76.67% | -6.66 pp |
| Precision@5 | — | 62.22% | — |
| Avg Inference Time | 11.41s | — | — |
| Success Rate | 100% | 100% | — |

**Key finding**: RAG context retrieval improved diagnosis accuracy by 40 percentage points, reducing misdiagnosed cases from 7 to 1.

---

## Disease Coverage (RAG System)

| Category | Result | Cases |
|---|---|---|
| Hemorrhagic Fevers | 3/3 | Ebola, Zika, Dengue |
| Bacterial Infections | 3/3 | Leptospirosis, Meningitis, Tuberculosis |
| Viral Infections | 5/5 | Malaria, Chikungunya, Yellow Fever, Typhoid, HIV/AIDS |
| Parasitic Diseases | 3/4 | Visceral Leishmaniasis, Schistosomiasis, Strongyloidiasis |
| Parasitic Diseases | 1/4 (missed) | Cutaneous Leishmaniasis (returned Buruli Ulcer) |

---

## How to View Results

### View Raw JSON
```bash
cat baseline_gemma_20260601_121422.json | jq .
cat rag_system_20260601_123213.json | jq .
```

### Extract Summary
```bash
cat rag_system_20260601_123213.json | jq '{accuracy: .diagnosis_accuracy_percent, keywords: .avg_keyword_match_percent, precision: .precision_at_5}'
```

### View Specific Test
```bash
cat rag_system_20260601_123213.json | jq '.results[0]'
```

---

## Reading the Results

Each test result includes:
- `test_index`: Test number (0–14)
- `query`: Patient presentation
- `expected_diagnosis`: Correct diagnosis
- `response`: Model's full response
- `extracted_diagnosis`: Parsed diagnosis from response
- `keyword_match`: Percentage of expected keywords found
- `diagnosis_correct`: Whether diagnosis matches expected
- `inference_time`: How long the request took
- `expected_keywords`: Expected medical terms to find
- `error`: Any errors that occurred (usually `"None"`)

---

## Running Tests

```bash
# Run baseline only
python tests/run_performance_tests.py --baseline

# Run RAG evaluation only
python tests/run_performance_tests.py --rag

# Run full comparison (baseline + RAG)
python tests/run_performance_tests.py --compare
```

Results are saved automatically to `tests/results/` with timestamps.

---

## Processing Results in Python

```python
import json

# Load and compare both result files
with open('baseline_gemma_20260601_121422.json') as f:
    baseline = json.load(f)

with open('rag_system_20260601_123213.json') as f:
    rag = json.load(f)

baseline_acc = float(baseline['diagnosis_accuracy_percent'])
rag_acc = float(rag['diagnosis_accuracy_percent'])

print(f"Baseline Accuracy : {baseline_acc:.2f}%")
print(f"RAG Accuracy      : {rag_acc:.2f}%")
print(f"Improvement       : +{rag_acc - baseline_acc:.2f} pp")

# Per-test breakdown
for b, r in zip(baseline['results'], rag['results']):
    status = "OK" if r.get('diagnosis_correct') == 'True' else "MISS"
    print(f"{status} {r['expected_diagnosis']}")
```

### Batch Compare Multiple Result Files

```python
import json, glob

for file in glob.glob('*.json'):
    with open(file) as f:
        data = json.load(f)
    print(f"{file}: {float(data['diagnosis_accuracy_percent']):.2f}% accuracy")
```

---

## Archiving Results

```bash
mkdir -p results/archive
cp results/*.json results/archive/
```

---

## References

- `tests/PERFORMANCE_TESTING.md` — Testing guide
- `BASELINE_ANALYSIS_REPORT.md` — Detailed baseline analysis
- `DOCUMENTATION_INDEX.md` — Complete project reference