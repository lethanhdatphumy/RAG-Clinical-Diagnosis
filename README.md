# Clinical RAG Pipeline

A clinical case–focused Retrieval-Augmented Generation (RAG) pipeline for working with medical case reports. The project handles the full lifecycle from raw PDFs to a searchable vector store and answer generation.

## Features

- **PDF ingestion & extraction** (`src/extraction/pdf_extractor.py`)
- **Cleaning & filtering** of extracted texts (`src/filtering/`)
- **Embeddings & vector indexing** using FAISS (`src/embedding/`, `data/vector/`, `data/vector_store/`)
- **Config‑driven pipeline** via `config/settings.py`
- **Evaluation scripts** for RAG quality (`tests/evaluate_rag.py`, `tests/ground_truth.py`)

## Project Structure

```text
Clinical/
├── main.py                # Entry point for running the pipeline / app
├── requirements.txt       # Python dependencies
├── config/
│   └── settings.py        # Global configuration
├── data/
│   ├── raw/               # Raw clinical case reports (e.g., PDFs)
│   ├── processed/         # Cleaned & structured text
│   ├── vector/            # Embedding artifacts (e.g., FAISS indexes)
│   └── vector_store/      # Persisted vector store for retrieval
├── src/
│   ├── extraction/        # PDF / text extraction
│   ├── filtering/         # Quality filtering, LLM cleanup, etc.
│   ├── embedding/         # Embedding generation and helpers
│   ├── generation/        # RAG answer generation
│   └── indexing/          # Index building and search utilities
└── tests/                 # Evaluation & regression tests
```

## Getting Started

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows

pip install -r requirements.txt
```

### 2. Configure settings

Edit `config/settings.py` to point to your local data directories, model names, and API keys (if required, e.g., for Gemini / other LLM providers used in `src/filtering/gemini_client.py`). **Do not commit secrets**—use environment variables or a local `.env` file.

### 3. Prepare data

Place your raw case reports (PDF or text) in `data/raw/` (or the path configured in `settings.py`).

Then run the extraction and processing pipeline, for example:

```bash
python main.py --stage extract
python main.py --stage filter
python main.py --stage index
```

(Adjust to whatever CLI options `main.py` exposes; see the docstring/help in that file.)

### 4. Build the vector index

Once processed texts are available in `data/processed/`, build the FAISS index and vector store:

```bash
python main.py --stage embed
python main.py --stage build_index
```

This will populate `data/vector/` and `data/vector_store/clinical_faiss/`.

### 5. Run RAG queries

After the index is built, you can query the system with clinical questions:

```bash
python main.py --stage query --question "A 35-year-old patient with fever and rash..."
```

The pipeline will:

1. Embed the query
2. Retrieve the most relevant case reports
3. Generate an answer using the RAG generator in `src/generation/rag_generator.py`

## Evaluation

Use the utilities under `tests/` to evaluate RAG performance against curated ground truth.

Example:

```bash
python -m tests.evaluate_rag
```

`tests/ground_truth.py` can be extended with more clinical questions and reference answers.

## Development

- Follow PEP 8 style guidelines.
- Prefer small, testable functions.
- Add or update tests under `tests/` when changing core logic.

To run tests:

```bash
pytest
```

## Roadmap / Ideas

- Web or notebook front‑end for interactive querying
- Support for multiple embedding backends (OpenAI, local models, etc.)
- More robust de‑identification of PHI in case texts
- Advanced evaluation (e.g., faithfulness and factuality metrics)

## License

Add your preferred license here (e.g., MIT, Apache‑2.0).
