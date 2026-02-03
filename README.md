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

- **Diagnosis Accuracy**: 100% on test cases.
- **Keyword Match**: 73.10%.
- **Precision@5**: 61.90%.

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