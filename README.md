# Clinical Diagnosis RAG System

An advanced AI-powered Retrieval-Augmented Generation (RAG) system designed to assist in the clinical diagnosis of tropical and infectious diseases. This project leverages state-of-the-art technologies to process medical case reports, generate embeddings, and provide evidence-based diagnostic recommendations.

## Key Features

- **End-to-End RAG Pipeline**: From data extraction to diagnosis generation.
- **PDF Extraction**: Extracts text and images from medical case reports.
- **LLM-Based Filtering**: Utilizes Google Gemini for structured data extraction.
- **Semantic Search**: Powered by Qdrant Cloud vector store for efficient retrieval.
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

## Tech Stack

| Component            | Technology                               |
|----------------------|------------------------------------------|
| **LLM**             | Google Gemini (Gemma-4-26B)              |
| **Embeddings**      | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Store**    | Qdrant Cloud                             |
| **Framework**       | LangChain                                |
| **Data Versioning** | DVC + AWS S3                             |
| **Web Interface**   | Streamlit                                |
| **Evaluation**      | Custom metrics framework                 |

## Data Preprocessing Pipeline

The system uses a 3-stage preprocessing pipeline to convert raw PDFs into searchable vectors:

```
data/raw/case_reports/*.pdf
          │
          ▼ Stage 1 — PDF Extraction
data/processed/extracted/<case>/metadata.json + images
          │
          ▼ Stage 2 — LLM Filtering (Gemini)
data/processed/filtered/<case>_filtered.json
  { diseases, symptoms, treatments, pathogens, lab_findings, ... }
          │
          ▼ Stage 3 — Embed & Upload
Qdrant Cloud — collection: clinical_cases (100 cases, 384-dim vectors)
```

**Key design decision**: instead of traditional text chunking, each PDF page is sent to Gemini which extracts structured clinical entities (diseases, symptoms, risk factors, etc.). This produces clean, semantically meaningful embeddings without PDF noise.

## Vector Store — Qdrant Cloud

As of October 2026, the project migrated from a local FAISS index to **Qdrant Cloud** for scalable, persistent vector storage.

### What's stored per case

Each of the 100 clinical cases is stored as a point with:
- A 384-dimensional cosine-similarity vector (all-MiniLM-L6-v2)
- Full payload: `case_id`, `diseases`, `symptoms`, `treatments`, `pathogens`, `laboratory_findings`, `vital_signs`, `risk_factors`, `patient_history`, `page_content`

### Uploading data to Qdrant

```bash
python -m src.embedding.qdrant_uploader
```

Required environment variables:
```
QDRANT_HOST=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION_NAME=clinical_cases   # optional
EMBEDDING_DIM=384                        # optional, fixed by model
EMBEDDING_BATCH_SIZE=32                  # optional
```

## Fine-Tuned Models

We have developed fine-tuned language models optimized for clinical diagnosis of tropical and infectious diseases.

### Available Models

- **Gemma 2 9B (Fine-tuned)**: A specialized version of Google's Gemma 2 9B model fine-tuned on clinical case data for tropical disease diagnosis.
- **Format**: GGUF (Quantized Q4_K_M) for efficient deployment and inference.

### Access Fine-Tuned Models

**Models Repository**: [Fine-Tuned Models](https://drive.google.com/drive/folders/1TUMCgApLyINMNutzZ67lwYRDrLJ5ix2M)

Fine-tuning was performed using LoRA (Low-Rank Adaptation) with the Unsloth framework.

## Getting Started

### Prerequisites

- Python 3.11+
- Qdrant Cloud account (or local Qdrant instance)
- Google Cloud API key for Gemini
- AWS account for S3 storage (data versioning)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/lethanhdatphumy/RAG-Clinical-Diagnosis.git
   cd RAG-Clinical-Diagnosis
   ```

2. **Install Dependencies**:
   ```bash
   uv add -r requirements.txt
   # or
   pip install -r requirements.txt
   ```

3. **Configure Credentials** — create a `.env` file:
   ```
   GOOGLE_API_KEY=your-gemini-api-key
   AWS_ACCESS_KEY_ID=your-aws-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret
   QDRANT_HOST=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your-qdrant-api-key
   ```

4. **Pull Data**:
   ```bash
   aws configure
   dvc pull
   ```

5. **Upload data to Qdrant** (first time only):
   ```bash
   python -m src.embedding.qdrant_uploader
   ```

6. **Run the Application**:
   ```bash
   # Web interface
   streamlit run app.py

   # Command line
   python main.py --stage query --question "Patient with fever and malaria symptoms..."
   ```

### Running the Full Pipeline (from scratch)

```bash
python main.py --stage extract    # Extract PDFs
python main.py --stage filter     # Filter with Gemini
python -m src.embedding.qdrant_uploader  # Embed & upload to Qdrant
```

### Docker Deployment

```bash
docker build -t clinical-rag:latest .

docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your-gemini-api-key \
  -e QDRANT_HOST=your-qdrant-url \
  -e QDRANT_API_KEY=your-qdrant-key \
  clinical-rag:latest
```

## Project Structure

```text
clinical-diagnosis-rag/
├── app.py                          # Streamlit web interface
├── main.py                         # Pipeline entry point
├── requirements.txt                # Python dependencies
├── config/
│   └── settings.py                # Configuration & API keys
├── data/                           # Data directories (tracked by DVC)
│   ├── raw/
│   ├── processed/
│   │   ├── extracted/             # Raw text + images from PDFs
│   │   └── filtered/              # Structured JSON per case
│   └── vector_store/              # Legacy FAISS (replaced by Qdrant)
├── src/
│   ├── extraction/                # PDF processing
│   ├── filtering/                 # Gemini-based entity extraction
│   ├── embedding/                 # Embeddings + Qdrant uploader
│   ├── generation/                # RAG query system (Qdrant-backed)
│   └── indexing/                  # Index utilities
└── tests/                          # Evaluation framework
    ├── test_gemma_baseline.py
    ├── evaluate_rag.py
    ├── run_performance_tests.py
    └── results/
```

## Evaluation

```bash
# Baseline (without RAG)
python tests/test_gemma_baseline.py

# RAG system evaluation
python tests/evaluate_rag.py

# Full comparison
python tests/run_performance_tests.py
```

Results are saved to `tests/results/` with timestamps.

## Roadmap

### Completed
- [x] End-to-end RAG pipeline (extract → filter → embed → query)
- [x] LLM-based structured entity extraction (Gemini)
- [x] Migrate vector store from FAISS to **Qdrant Cloud**
- [x] `QdrantUploader` class with batched upsert
- [x] Fine-tuned Gemma 2 9B (LoRA + Unsloth)
- [x] Docker containerization

### In Progress
- [ ] **LangGraph agentic routing** — a router agent that decides whether to use RAG (for specific patient cases) or the fine-tuned LLM (for general disease knowledge questions)

### Planned
- [ ] Deploy to cloud (Google Cloud Run / AWS Lambda)
- [ ] Differential diagnosis (top-3 predictions)
- [ ] Multi-language support
- [ ] Fine-tune embeddings on medical corpus
- [ ] User feedback loop for model improvement
- [ ] Migrate to PostgreSQL + pgvector
- [ ] Comprehensive pytest suite

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add tests for new features.
4. Submit a pull request.

## Contact

- GitHub: [@Thanh Dat Le](https://github.com/lethanhdatphumy)
- LinkedIn: [Thanh Dat Le](https://www.linkedin.com/in/thanh-dat-le-a9221125b/)

---

**If you find this project useful, please star the repository!**
