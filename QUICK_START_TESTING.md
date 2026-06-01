# Quick Start: Baseline Performance Testing

## TL;DR - Get Results in 5 Minutes

### Step 1: Run Baseline Test
```bash
cd /Users/admin/PycharmProjects/Clinical
python tests/test_gemma_baseline.py
```

### Step 2: View Results
Results are automatically saved to `tests/results/baseline_gemma_*.json`

### Step 3: Compare with RAG (Optional)
```bash
python tests/run_performance_tests.py
# Answer "yes" when prompted to run RAG test
```

## What You Get

### Baseline Metrics (Gemma without RAG)
- **Diagnosis Accuracy**: How many diseases correctly identified
- **Keyword Match**: Medical terminology coverage
- **Inference Time**: Speed per query
- **Error Rate**: System reliability

### Comparison Metrics (if running RAG test)
- Side-by-side performance comparison
- Percentage improvement from RAG
- Statistical summary

## Expected Output

```
════════════════════════════════════════════════════════════════════════════════
TEST 1/15
════════════════════════════════════════════════════════════════════════════════

Query: 20-year-old Sudanese refugee in Uganda with fever...
Expected Diagnosis: ebola

Model Response:
[Full response from Gemma model]

Extracted Diagnosis: ebola
Inference Time: 1.45s
Keyword Match: 80.0%
Diagnosis Correct: True

...

════════════════════════════════════════════════════════════════════════════════
EVALUATION SUMMARY - GEMMA BASELINE
════════════════════════════════════════════════════════════════════════════════

Timestamp: 2024-01-15T10:30:45
Model: models/gemma-3-27b-it

Test Results:
  Total Tests:        15
  Successful Tests:   15
  Failed Tests:       0
  Error Rate:         0.00%

Performance Metrics:
  Avg Inference Time: 1.42s
  Keyword Match Rate: 72.50%
  Diagnosis Accuracy: 60.00%

════════════════════════════════════════════════════════════════════════════════
```

## Files Created for Testing

```
/tests/
├── test_gemma_baseline.py          # Baseline evaluation (NEW)
├── run_performance_tests.py         # Test orchestration (NEW)
├── PERFORMANCE_TESTING.md           # Detailed docs (NEW)
├── results/                         # Results storage (NEW)
│   ├── baseline_gemma_*.json       # Baseline results
│   ├── rag_system_*.json           # RAG results (if run)
│   └── comparison_report_*.json    # Comparison (if run)
├── evaluate_rag.py                 # RAG evaluation (existing)
├── ground_truth.py                 # Test cases (existing)
└── __init__.py
```

## Key Metrics Explained

### Diagnosis Accuracy
- What % of diseases did the model identify correctly
- Important for: Clinical reliability
- Baseline expected: 50-60%
- RAG expected: 70-75%

### Keyword Match Rate
- What % of medical terminology did the model use
- Important for: Medical knowledge depth
- Baseline expected: 60-75%
- RAG expected: 75-85%

### Inference Time
- How many seconds per query
- Important for: Real-time clinical use
- Baseline expected: 1.2-1.5s
- RAG expected: 2.0-2.5s (slightly slower, but more accurate)

### Error Rate
- What % of queries failed
- Important for: System reliability
- Expected: < 5%

## Troubleshooting

**API Key Error?**
```bash
# Create .env file with your API key
echo "GOOGLE_API_KEY=your-api-key" > .env
```

**Vector store missing?**
```bash
# Pull data from DVC
dvc pull
```

**Connection timeout?**
- Check internet connection
- May take 30-60 seconds per query if network is slow

## Next Steps

1. ✅ Run baseline test
2. ✅ Note the results
3. ✅ Run RAG comparison to see improvement
4. ✅ Download fine-tuned models from [Google Drive](https://drive.google.com/drive/folders/1TUMCgApLyINMNutzZ67lwYRDrLJ5ix2M)
5. ✅ Test fine-tuned models to see if they improve accuracy

## More Information

- **Detailed Testing Guide**: See `tests/PERFORMANCE_TESTING.md`
- **Implementation Details**: See `BASELINE_TESTING_SETUP.md`
- **RAG System**: See `src/generation/rag_generator.py`
- **Configuration**: See `config/settings.py`

---

**Ready to test?** Run this command now:
```bash
python tests/test_gemma_baseline.py
```
