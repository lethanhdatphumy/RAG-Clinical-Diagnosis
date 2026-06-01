"""
RAG System Evaluation Module

Evaluates the Retrieval-Augmented Generation system on tropical disease diagnosis.
Measures accuracy, keyword match rate, and precision@K metrics.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import List, Dict, Tuple
from src.generation.rag_generator import ClinicalRAG
from tests.ground_truth import GROUND_TRUTH
from datetime import datetime
from config.settings import Config


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_all_test(rag_system: ClinicalRAG, test_cases: List[Dict]) -> List[Dict]:
    """
    Run RAG system on all test cases.

    Args:
        rag_system: ClinicalRAG instance
        test_cases: List of test case dictionaries with 'query', 'expected_diagnosis', and 'expected_keywords'

    Returns:
        List of response dictionaries with actual responses and retrieved cases
    """
    responses = []

    for i, test_case in enumerate(test_cases):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i + 1}/{len(test_cases)}: {test_case['expected_diagnosis'].upper()}")
        print(f"{'=' * 80}")
        print(f"Query: {test_case['query'][:100]}...")

        try:
            result = rag_system.query(test_case['query'])

            response = {
                "test_index": i,
                "test_case": test_case['query'],
                "expected_diagnosis": test_case['expected_diagnosis'],
                "expected_keywords": test_case['expected_keywords'],
                "actual_response": result["result"].lower(),
                "retrieved_cases": [doc.metadata['case_id'] for doc in result['source_documents']],
                "num_retrieved": len(result['source_documents']),
                "error": None
            }
            responses.append(response)
            print(f"✓ Success - Retrieved {len(result['source_documents'])} cases")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            response = {
                "test_index": i,
                "test_case": test_case['query'],
                "expected_diagnosis": test_case['expected_diagnosis'],
                "expected_keywords": test_case['expected_keywords'],
                "actual_response": "",
                "retrieved_cases": [],
                "num_retrieved": 0,
                "error": str(e)
            }
            responses.append(response)

    return responses


def calculate_diagnosis_accuracy(responses: List[Dict]) -> Tuple[float, int, int]:
    """
    Calculate diagnosis accuracy by checking if expected diagnosis appears in actual response.

    Args:
        responses: List of response dictionaries

    Returns:
        Tuple of (accuracy_rate, correct_count, total_count)
    """
    if not responses:
        return 0.0, 0, 0

    correct = sum(
        1 for response in responses
        if response['error'] is None and response['expected_diagnosis'].lower() in response['actual_response'].lower()
    )

    total = len([r for r in responses if r['error'] is None])
    accuracy = correct / total if total > 0 else 0.0

    return accuracy, correct, total


def calculate_match_keywords(responses: List[Dict]) -> float:
    """
    Calculate average keyword match rate across all responses.

    Args:
        responses: List of response dictionaries

    Returns:
        Average percentage of expected keywords found (0.0-1.0)
    """
    if not responses:
        return 0.0

    matches = []

    for response in responses:
        if response['error'] is None:
            matched = 0
            expected_keywords = response['expected_keywords']
            actual = response['actual_response']

            for expected_keyword in expected_keywords:
                if expected_keyword.lower() in actual.lower():
                    matched += 1

            if expected_keywords:
                matches.append(matched / len(expected_keywords))

    return sum(matches) / len(matches) if matches else 0.0


def is_case_relevant(case_id: str, expected_disease: str, expected_keywords: List[str]) -> bool:
    """
    Check if a case file is relevant to the expected disease and keywords.

    Args:
        case_id: The case identifier
        expected_disease: Expected disease name
        expected_keywords: List of expected medical keywords

    Returns:
        True if case contains expected disease or keywords, False otherwise
    """
    project_root = get_project_root()
    case_path = project_root / "data" / "processed" / "filtered" / f"{case_id}_filtered.json"

    if not case_path.exists():
        return False

    try:
        with open(case_path, 'r') as f:
            case_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    diseases_text = " ".join(case_data.get('diseases', [])).lower()
    history_text = case_data.get('patient_history', '').lower()
    symptoms_text = " ".join(case_data.get('symptoms', [])).lower()

    combined_text = diseases_text + " " + history_text + " " + symptoms_text

    # Check if expected disease is in combined text
    if expected_disease.lower() in combined_text:
        return True

    # Check if any expected keywords are in combined text
    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            return True

    return False


def calculate_precision_at_k(results: List[Dict], k: int = 5) -> float:
    """
    Calculate average Precision@K metric.

    Precision@K measures the proportion of the top-K retrieved cases that are relevant
    to the expected diagnosis.

    Args:
        results: List of response dictionaries containing 'retrieved_cases' key
        k: Number of top results to consider (default: 5)

    Returns:
        Average precision@k across all test cases (0.0-1.0)
    """
    if not results:
        return 0.0

    precisions = []

    for result in results:
        if result['error'] is not None:
            continue

        # Get top K retrieved cases
        retrieved_cases = result.get('retrieved_cases', [])[:k]

        if not retrieved_cases:
            precisions.append(0.0)
            continue

        # Count relevant cases
        relevant_count = sum(
            1 for case_id in retrieved_cases
            if is_case_relevant(case_id, result['expected_diagnosis'], result['expected_keywords'])
        )

        # Calculate precision for this test case
        precision = relevant_count / len(retrieved_cases)
        precisions.append(precision)

    return sum(precisions) / len(precisions) if precisions else 0.0


if __name__ == "__main__":
    print(f"\n{'#' * 80}")
    print(f"RAG SYSTEM EVALUATION")
    print(f"{'#' * 80}\n")

    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag = ClinicalRAG()

        # Run all tests
        print(f"Running {len(GROUND_TRUTH)} test cases...\n")
        results = run_all_test(rag, GROUND_TRUTH)

        # Calculate metrics
        accuracy, correct, total = calculate_diagnosis_accuracy(results)
        keyword_match = calculate_match_keywords(results)
        precision_at_5 = calculate_precision_at_k(results, k=5)

        # Print results
        print(f"\n{'#' * 80}")
        print(f"RAG SYSTEM RESULTS")
        print(f"{'#' * 80}\n")
        print(f"Diagnosis Accuracy: {correct}/{total} = {accuracy:.2%}")
        print(f"Keyword Match Rate: {keyword_match:.2%}")
        print(f"Precision@5: {precision_at_5:.2%}\n")

        # Prepare metrics dictionary
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": Config.GEMINI_MODEL,
            "embedding_model": Config.TEXT_EMBEDDING_MODEL,
            "evaluation_type": "RAG System",
            "total_tests": total,
            "successful_tests": total,
            "failed_tests": len([r for r in results if r['error'] is not None]),
            "diagnosis_accuracy_percent": round(accuracy * 100, 2),
            "keyword_match_percent": round(keyword_match * 100, 2),
            "precision_at_5_percent": round(precision_at_5 * 100, 2),
            "correct_diagnoses": correct,
            "results": results
        }

        # Save results to tests/results directory
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f"rag_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Serialize results
        def serialize_obj(obj):
            if isinstance(obj, list):
                return [serialize_obj(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_obj(v) for k, v in obj.items()}
            else:
                return str(obj)

        with open(results_file, 'w') as f:
            json.dump(serialize_obj(metrics), f, indent=2)

        print(f"Results saved to: {results_file}\n")
        print(f"{'#' * 80}\n")

    except Exception as e:
        print(f"\n✗ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

