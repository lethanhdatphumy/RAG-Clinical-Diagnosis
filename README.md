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