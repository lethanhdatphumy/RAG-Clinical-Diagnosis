import argparse
import sys
from pathlib import Path

from src.extraction.pdf_extractor import ClinicalPDFExtractor
from src.filtering.gemini_client import GeminiClient
from src.embedding.embedder import ClinicalEmbedder
from src.generation.rag_generator import ClinicalRAG
from config.settings import Config


def extract_stage():
    """Extract text and images from PDF case reports"""
    print("Starting PDF extraction...")
    extractor = ClinicalPDFExtractor()
    cases = extractor.extract_all_report()
    print(f"Extracted {len(cases)} case reports")


def filter_stage():
    """Filter and clean the extracted text using Gemini or Google LLM models."""
    print("Starting text filtering with Gemini...")
    client = GeminiClient()
    filtered_cases = client.process_all_cases()
    print(f"Filtered {len(filtered_cases)} cases")


def embed_stage():
    """Create embeddings and build FAISS vector store"""
    print("Creating embeddings and vector store...")
    embedder = ClinicalEmbedder()

    # Load filtered documents
    documents = embedder.load_filtered_as_document()

    # Create vector store
    vector_store = embedder.create_vector_store(documents, Config.TEXT_EMBEDDING_MODEL)

    # Save vector store
    embedder.save_vector_store(vector_store)
    print("Vector store created and saved")


def build_index_stage():
    """Build searchable index (alias for embed_stage)"""
    print("Building searchable index...")
    embed_stage()


def query_stage(question):
    """Query the RAG system"""
    print(f"Querying RAG system: {question}")

    rag = ClinicalRAG()
    result = rag.query(question)

    print("\n" + "="*80)
    print("DIAGNOSIS & RECOMMENDATIONS:")
    print("="*80)
    print(result['result'])

    print("\n" + "="*80)
    print("SIMILAR CASES RETRIEVED:")
    print("="*80)
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n{i}. Case: {doc.metadata['case_id']}")
        print(f"   Content: {doc.page_content[:200]}...")


def run_full_pipeline():
    """Run the complete pipeline from PDFs to RAG system"""
    print("Running full Clinical RAG pipeline...")

    extract_stage()
    filter_stage()
    embed_stage()

    # Test query
    test_question = "Patient with fever, bleeding, and recent travel to West Africa"
    query_stage(test_question)


def main():
    parser = argparse.ArgumentParser(
        description="Clinical RAG Pipeline - Medical Case Report Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage extract          # Extract PDFs
  python main.py --stage filter           # Filter with Gemini
  python main.py --stage embed            # Create embeddings
  python main.py --stage query --question "Patient with fever..."
  python main.py --stage full             # Run complete pipeline
        """
    )

    parser.add_argument(
        "--stage",
        choices=['extract', 'filter', 'embed', 'index', 'build_index', 'query', 'full'],
        required=True,
        help="Pipeline stage to run"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Clinical question for RAG query (required for 'query' stage)"
    )

    args = parser.parse_args()

    # Validate question for query stage
    if args.stage == 'query' and not args.question:
        parser.error("--question is required when using --stage query")

    # Execute based on stage
    try:
        if args.stage == 'extract':
            extract_stage()
        elif args.stage == 'filter':
            filter_stage()
        elif args.stage in ['embed', 'index', 'build_index']:
            embed_stage()
        elif args.stage == 'query':
            query_stage(args.question)
        elif args.stage == 'full':
            run_full_pipeline()

    except Exception as e:
        print(f"Error in {args.stage} stage: {e}")
        sys.exit(1)

    print(f"\n{args.stage.title()} stage completed successfully!")


if __name__ == "__main__":
    main()