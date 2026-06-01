# Clinical Diagnosis RAG System

An advanced AI-powered Retrieval-Augmented Generation (RAG) system designed to assist in the clinical diagnosis of tropical and infectious diseases. This project leverages state-of-the-art technologies to process medical case reports, generate embeddings, and provide evidence-based diagnostic recommendations.

## Key Features

- **End-to-End RAG Pipeline**: From data extraction to diagnosis generation.
- **PDF Extraction**: Extracts text and images from medical case reports.
- **LLM-Based Filtering**: Utilizes Google Gemini for structured data extraction.
- **Semantic Search**: Powered by FAISS vector store for efficient retrieval.
- **Interactive Web Interface**: Built with Streamlit for user-friendly diagnosis.
- **Comprehensive Evaluation Framework**: Includes metrics for accuracy and precision.
- **Data Versioning**: Managed with DVC and AWS S3 for reproducibility.

## Why This Project Matters

Tropical and infectious diseases often require timely and accurate diagnosis. This system aims to:
- Enhance diagnostic accuracy with AI-driven insights.
- Provide a scalable solution for medical professionals.
- Facilitate research and education in tropical medicine.

## Performance Highlights

### Baseline Performance (Gemma-4-26B without RAG)
**Test Date**: June 1, 2026 | **Model**: models/gemma-4-26b-a4b-it
- **Diagnosis Accuracy**: 53.33%
- **Keyword Match Rate**: 83.33%
- **Average Inference Time**: 11.41 seconds
- **Error Rate**: 0.00% (15/15 successful tests)

### RAG System Performance (Gemma-4-26B with RAG)
**Test Date**: June 1, 2026 | **Model**: models/gemma-4-26b-a4b-it
- **Diagnosis Accuracy**: 93.33% ⭐ (+40% improvement)
- **Keyword Match Rate**: 76.67%
- **Precision@5**: 62.22%
- **Error Rate**: 0.00% (15/15 successful tests)
- **Embedding Model**: all-MiniLM-L6-v2

**Key Finding**: RAG system shows significant accuracy improvement of **40 percentage points** (from 53.33% to 93.33%) by providing contextual case examples for diagnosis support.

### Detailed Performance Comparison

| Metric | Baseline (No RAG) | RAG System | Improvement |
|--------|-------------------|-----------|-------------|
| **Diagnosis Accuracy** | 53.33% (8/15) | 93.33% (14/15) | **+40.00%** ⭐ |
| **Keyword Match Rate** | 83.33% | 76.67% | -6.66% |
| **Precision@5** | N/A | 62.22% | - |
| **Test Success Rate** | 100% (15/15) | 100% (15/15) | Maintained |
| **Error Rate** | 0% | 0% | Maintained |

**Analysis**: The RAG system demonstrates exceptional performance improvement in diagnostic accuracy. By retrieving contextual case examples and incorporating them into the diagnosis reasoning process, the system achieves a 40 percentage point improvement over the baseline. This validates the core hypothesis that retrieval-augmented generation significantly enhances medical AI decision-making.

## Tech Stack

| Component         | Technology                               |
|-------------------|------------------------------------------|
| **LLM**          | Google Gemini (Gemma)                    |
| **Embeddings**   | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS                                    |
| **Framework**    | LangChain                                |
| **Data Versioning** | DVC + AWS S3                             |
| **Web Interface** | Streamlit                                |
| **Evaluation**   | Custom metrics framework                 |

## Fine-Tuned Models

We have developed fine-tuned language models optimized for clinical diagnosis of tropical and infectious diseases. These models are stored in our Google Drive repository and can be accessed for improved diagnostic accuracy.

### Available Models

- **Gemma 2 9B (Fine-tuned)**: A specialized version of Google's Gemma 2 9B model fine-tuned on clinical case data for tropical disease diagnosis.
- **Format**: GGUF (Quantized Q4_K_M) for efficient deployment and inference.

### Access Fine-Tuned Models

The fine-tuned models are available in the following Google Drive folder:
**Models Repository**: [Fine-Tuned Models](https://drive.google.com/drive/folders/1TUMCgApLyINMNutzZ67lwYRDrLJ5ix2M)

### Using the Fine-Tuned Models

1. **Download the Model**:
   - Visit the [Google Drive folder](https://drive.google.com/drive/folders/1TUMCgApLyINMNutzZ67lwYRDrLJ5ix2M)
   - Download the GGUF file and Modelfile

2. **Integration**:
   - The fine-tuned models can be loaded via Ollama or LM Studio for local inference
   - Update configuration in `config/settings.py` to point to your local model

3. **Performance Benefits**:
   - Improved diagnostic accuracy on tropical disease cases
   - Better contextual understanding of clinical presentations
   - Reduced latency for inference when running locally

**Note**: Fine-tuning was performed using LoRA (Low-Rank Adaptation) with the Unsloth framework for efficient training on clinical case data.

## Baseline Performance Testing

To evaluate the base Gemma model's diagnostic capability without RAG augmentation, we provide a comprehensive baseline performance test. This helps measure the improvement gained by using the RAG system.

### Running the Baseline Test

```bash
# Run the baseline performance evaluation
python tests/test_gemma_baseline.py
```

### What the Baseline Test Does

The baseline test (`tests/test_gemma_baseline.py`) evaluates the Gemma model's ability to:
- Diagnose 15 different tropical and infectious diseases
- Generate relevant medical keywords and evidence
- Provide diagnostic accuracy without retrieval-augmented generation

### Metrics Evaluated

- **Diagnosis Accuracy**: Percentage of correct primary diagnoses
- **Keyword Match Rate**: Average percentage of expected medical keywords mentioned
- **Inference Time**: Average response time per query
- **Error Rate**: Number of failed queries

### Sample Ground Truth Test Cases

The baseline test includes diverse clinical presentations:
- Hemorrhagic diseases (Ebola, Dengue, Yellow Fever)
- Parasitic infections (Malaria, Leishmaniasis, Schistosomiasis)
- Bacterial infections (Tuberculosis, Typhoid, Meningitis)
- Viral infections (HIV, Zika, Chikungunya)
- Helminth infections (Strongyloidiasis)

### Comparing Baseline vs RAG Performance

After running the baseline test, compare the results with the RAG system performance:

```bash
# Run the comparison script - it will prompt you to run RAG test too
python tests/run_performance_tests.py

# Or test the RAG system separately
python tests/evaluate_rag.py

# Compare metrics:
# - Baseline accuracy vs RAG accuracy
# - Baseline keyword match vs RAG keyword match  
# - Impact of retrieval-augmented generation
```

This comparison demonstrates the value of including contextual case reports in the diagnosis generation process.

### Test Results & Evaluation

The performance metrics above were obtained from comprehensive testing on 15 diverse tropical disease cases:

**Baseline Test Results**:
- File: `tests/results/baseline_gemma_20260601_121422.json`
- Date: June 1, 2026
- Test Cases: 15 (Hemorrhagic fevers, Parasitic diseases, Bacterial infections, Viral infections)

**RAG System Test Results**:
- File: `tests/results/rag_system_20260601_123213.json`
- Date: June 1, 2026
- Test Cases: 15 (Same disease categories for fair comparison)

**Evaluation Metrics**:
- Diagnosis Accuracy: Percentage of correct primary diagnoses
- Keyword Match: Percentage of expected medical terminology present
- Precision@5: Relevance of top-5 retrieved case examples
- Test Success Rate: Percentage of tests completed without errors

### Running Tests

To evaluate the system:

```bash
# Run baseline test (without RAG)
python tests/test_gemma_baseline.py

# Run RAG system evaluation
python tests/evaluate_rag.py

# Run full comparison
python tests/run_performance_tests.py
```

Results are automatically saved to `tests/results/` with timestamps for tracking performance over time.

## Getting Started

### Prerequisites

- Python 3.11+
- AWS account for S3 storage.
- Google Cloud API key for Gemini.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/clinical-diagnosis-rag.git
   cd clinical-diagnosis-rag
   ```

2. **Set Up Environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Configure Credentials**:
   Create a `.env` file with your API keys:
   ```bash
   GOOGLE_API_KEY=your-gemini-api-key
   AWS_ACCESS_KEY_ID=your-aws-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret
   ```

4. **Pull Data**:
   ```bash
   aws configure
   dvc pull
   ```

5. **Run the Application**:
   - **Web Interface**:
     ```bash
     streamlit run app.py
     ```
   - **Command Line**:
     ```bash
     python main.py --stage query --question "Patient with fever and malaria symptoms..."
     ```

### Docker Deployment

#### Prerequisites for Docker
- Docker installed and running
- Docker Compose (optional, for advanced setups)

#### Build and Run with Docker

1. **Build the Docker Image**:
   ```bash
   docker build -t clinical-rag:latest .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8501:8501 \
     -e GOOGLE_API_KEY=your-gemini-api-key \
     -e AWS_ACCESS_KEY_ID=your-aws-key \
     -e AWS_SECRET_ACCESS_KEY=your-aws-secret \
     clinical-rag:latest
   ```

3. **Access the Application**:
   Open your browser and navigate to `http://localhost:8501`

#### Advanced Docker Options

**Run with Volume Mounting** (for local data access):
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/Clinical/data \
  -e GOOGLE_API_KEY=your-gemini-api-key \
  clinical-rag:latest
```

**Run in Detached Mode** (background process):
```bash
docker run -d -p 8501:8501 \
  --name clinical-rag \
  -e GOOGLE_API_KEY=your-gemini-api-key \
  clinical-rag:latest
```

**View Container Logs**:
```bash
docker logs clinical-rag
```

**Stop the Container**:
```bash
docker stop clinical-rag
```

## Project Structure

```text
clinical-diagnosis-rag/
├── app.py                      # Streamlit web interface
├── main.py                     # Pipeline entry point
├── requirements.txt            # Python dependencies
├── config/
│   └── settings.py            # Configuration & API keys
├── data/                       # Data directories (tracked by DVC)
│   ├── raw/
│   ├── processed/
│   └── vector_store/
├── src/
│   ├── extraction/            # PDF processing
│   ├── filtering/             # Data filtering
│   ├── embedding/             # Vector embeddings
│   ├── generation/            # RAG query system
│   └── indexing/              # Index utilities
└── tests/                      # Evaluation framework
```

## Roadmap

- [ ] Deploy to cloud (Google Cloud Run / AWS Lambda).
- [ ] Add differential diagnosis (top-3 predictions).
- [ ] Multi-language support.
- [ ] Fine-tune embeddings on medical corpus.
- [ ] User feedback loop for model improvement.
- [ ] Migrate to PostgreSQL + pgvector.
- [ ] Add comprehensive test suite with pytest.
- [ ] Docker containerization.

## Contributing

We welcome contributions! To get started:
1. Fork the repository.
2. Create a feature branch.
3. Add tests for new features.
4. Submit a pull request.

## Contact

For questions or collaboration:
- GitHub: [@Thanh Dat Le](https://github.com/lethanhdatphumy)
- LinkedIn: [Thanh Dat Le](https://www.linkedin.com/in/thanh-dat-le-a9221125b/)

---

**If you find this project useful, please star the repository!**