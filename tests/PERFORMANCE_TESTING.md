# Performance Testing Guide

## Overview

This directory contains comprehensive performance evaluation tools for the Clinical Diagnosis System. The tests allow you to:

1. **Baseline Test**: Evaluate the Gemma model's diagnostic capability **without RAG**
2. **RAG Test**: Evaluate the Gemma model's performance **with RAG** (retrieval-augmented generation)
3. **Comparison**: Compare the two approaches to measure RAG's impact on accuracy

## Quick Start

### Run Only Baseline Test
```bash
python tests/test_gemma_baseline.py
```

This will:
- Test the base Gemma model on 15 tropical disease cases
- Evaluate diagnosis accuracy, keyword matching, and inference time
- Save results to `baseline_gemma_results.json`

### Run Full Evaluation Suite
```bash
python tests/run_performance_tests.py
```

This will:
- Run baseline test
- Ask if you want to run RAG test
- Generate a comparison report showing improvements

### Run Only RAG Test
```bash
python tests/evaluate_rag.py
```

This will:
- Test the RAG system on the same 15 disease cases
- Show how retrieval-augmented generation improves accuracy
- Save results to `rag_results.json`

## Understanding the Results

### Baseline Test Output

The baseline test reports:
- **Diagnosis Accuracy**: % of correct primary diagnoses (without context)
- **Keyword Match Rate**: % of expected medical keywords mentioned on average
- **Average Inference Time**: Time taken per query in seconds
- **Error Rate**: % of queries that failed

### Comparison Report

When comparing baseline vs RAG, you'll see:
- Side-by-side metrics for both approaches
- Absolute improvements (e.g., +15% accuracy)
- Percentage improvements (e.g., +20% faster)

Example:
```
Metric                          Baseline        RAG             Improvement
─────────────────────────────────────────────────────────────────────────────
Diagnosis Accuracy (%)              45.00%      61.90%              +16.90%
Keyword Match Rate (%)              60.00%      73.10%              +13.10%
Avg Inference Time (s)               1.50s       2.10s               +0.60s
```

## Test Cases

Both tests use 15 diverse tropical disease cases:

1. **Ebola** - Hemorrhagic fever during outbreak
2. **Visceral Leishmaniasis** - Kala-azar with splenomegaly
3. **Leptospirosis** - Weil's disease after contaminated water exposure
4. **Zika Virus** - Birth defect risk in pregnancy
5. **Malaria** - Cyclical fever with splenomegaly
6. **Tuberculosis** - Cavitary lung disease
7. **Meningitis** - Bacterial CNS infection
8. **Chikungunya** - Alphavirus arthralgia
9. **Yellow Fever** - Hepatic hemorrhagic fever
10. **Typhoid Fever** - Enteric fever from contaminated food
11. **HIV/AIDS** - Immunodeficiency with opportunistic infections
12. **Cutaneous Leishmaniasis** - Sandfly-transmitted skin ulcer
13. **Dengue Fever** - Hemorrhagic dengue with thrombocytopenia
14. **Schistosomiasis** - Parasitic infection from freshwater
15. **Strongyloidiasis** - Helminth hyperinfection syndrome

## Results Storage

All results are stored in `tests/results/` directory:

- `baseline_gemma_YYYYMMDD_HHMMSS.json` - Baseline test detailed results
- `rag_system_YYYYMMDD_HHMMSS.json` - RAG system detailed results
- `comparison_report_YYYYMMDD_HHMMSS.json` - Side-by-side comparison

## Key Metrics Explained

### Diagnosis Accuracy
- **Definition**: % of queries where the model correctly identified the primary diagnosis
- **Importance**: Direct measure of diagnostic correctness
- **Baseline vs RAG**: RAG should improve this by providing contextual cases

### Keyword Match Rate
- **Definition**: % of expected medical keywords the model mentioned on average
- **Importance**: Shows depth of medical knowledge demonstrated
- **Baseline vs RAG**: RAG should improve by retrieving keyword-rich case reports

### Inference Time
- **Definition**: Average seconds to generate a response per query
- **Importance**: Critical for real-time clinical decision support
- **Trade-off**: RAG may increase time due to retrieval step

### Error Rate
- **Definition**: % of queries that resulted in API/processing errors
- **Importance**: System reliability indicator
- **Acceptable Range**: Should be < 5%

## Interpreting RAG Improvements

### Expected Improvements
- **Accuracy**: +15-25% (contextual cases help diagnosis)
- **Keyword Match**: +10-20% (case reports contain relevant medical terms)
- **Error Rate**: Should remain stable or improve

### Possible Trade-offs
- **Inference Time**: May increase 1.2-1.5x (retrieval adds time)
- **This is acceptable**: The accuracy improvements outweigh latency cost

## Fine-Tuned Models

After establishing baseline and RAG performance, you can test fine-tuned models:

1. Download fine-tuned Gemma from: [Google Drive](https://drive.google.com/drive/folders/1TUMCgApLyINMNutzZ67lwYRDrLJ5ix2M)
2. Configure local model in `config/settings.py`
3. Run tests again to compare with Google Gemini API version

## Customizing Tests

### Adding More Test Cases
Edit `tests/test_gemma_baseline.py` and modify `GEMMA_BASELINE_TESTS` list:

```python
GEMMA_BASELINE_TESTS = [
    {
        "query": "Your new test case...",
        "expected_diagnosis": "disease_name",
        "expected_keywords": ["keyword1", "keyword2", ...]
    },
    # ... more cases
]
```

### Adjusting Model Parameters
Edit `config/settings.py`:

```python
# Temperature: 0.0 (deterministic) to 1.0 (creative)
# Higher temp = more varied responses
temperature=0.7

# Max output tokens: Controls response length
max_output_tokens=512
```

### Changing Retrieval Parameters
For RAG test, edit `config/settings.py`:

```python
# Number of cases to retrieve
TOP_K_RETRIEVAL = 3  # Increase for more context
```

## Troubleshooting

### "GOOGLE_API_KEY not set"
```bash
# Create .env file in project root
echo "GOOGLE_API_KEY=your-api-key" > .env
echo "AWS_ACCESS_KEY_ID=your-aws-key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your-aws-secret" >> .env
```

### "Vector store not found"
```bash
# Pull data from DVC
dvc pull
```

### Rate Limit Errors
- Tests make sequential API calls
- If hitting rate limits, add delays in test files
- Consider using batch inference or fine-tuned local models

### Memory Issues with Large Models
- Use quantized/fine-tuned models from Google Drive
- They're optimized for local deployment with lower memory footprint

## Next Steps

1. **Run baseline test** to establish base model performance
2. **Run RAG test** to measure improvement with context
3. **Review comparison report** to validate RAG benefits
4. **Experiment with fine-tuned models** from Google Drive
5. **Deploy best-performing configuration** to production

## Contributing

To add new test cases or metrics:
1. Update `GEMMA_BASELINE_TESTS` or `GROUND_TRUTH` 
2. Add new metrics to the evaluator classes
3. Document the changes in this file

---

For detailed implementation, see:
- `test_gemma_baseline.py` - Baseline evaluation logic
- `run_performance_tests.py` - Test orchestration
- `evaluate_rag.py` - RAG system evaluation
- `ground_truth.py` - Test case definitions
