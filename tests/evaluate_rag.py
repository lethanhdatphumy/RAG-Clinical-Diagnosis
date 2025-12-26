import json
from pathlib import Path
from src.generation.rag_generator import ClinicalRAG
from tests.ground_truth import GROUND_TRUTH
from datetime import datetime
from config.settings import Config


def get_project_root():
    return Path(__file__).parent.parent


def run_all_test(rag_system, test_cases):
    """
    :param rag_system: ClinicalRag Instance
    :param test_cases: the dictionary contains the expected diagnosis and expected keywords
    :return: return the dictionary, includes the actual and expected diagnosis.
    """

    responses = []
    for i, test_case in enumerate(test_cases):
        response = {}
        print(f"\n{'=' * 80}")
        print(f"TESTING {i}")
        print(f"\n{'=' * 80}")

        result = rag_system.query(test_case['query'])

        response["test_case"] = test_case["query"]
        response["expected_diagnosis"] = test_case["expected_diagnosis"]
        response["expected_keywords"] = test_case["expected_keywords"]
        response["actual_response"] = result["result"].lower()
        response["retrieved"] = []

        for j, doc in enumerate(result['source_documents'], 1):
            response["retrieved"].append(doc.metadata['case_id'])
        responses.append(response)

        # Review the RAG
        # print(f"Expected: {test_case['expected_diagnosis']}")
        # print(f"Response preview: {short_response}")
        # print(f"Retrieved: {len(result['source_documents'])} cases")
    return responses


def calculate_diagnosis_accuracy(responses):
    correct = 0

    for response in responses:
        expected = response['expected_diagnosis']
        actual = response['actual_response']
        if expected.lower() in actual.lower():
            correct += 1
    accuracy = correct / len(responses)

    return accuracy, correct, len(responses)


def calculate_match_keywords(response):
    matches = []
    for response in response:
        matched = 0
        expected_keywords = response['expected_keywords']
        actual = response['actual_response']
        for expected_keyword in expected_keywords:
            if expected_keyword.lower() in actual.lower():
                matched += 1
        matches.append(matched / len(expected_keywords))
    return sum(matches) / len(matches)


def is_case_relevant(case_id, expected_disease, expected_keywords):
    # Use the absolute path
    project_root = get_project_root()
    case_path = project_root / "data" / "processed" / "filtered" / f"{case_id}_filtered.json"
    # Check if the path does not exist
    if not case_path.exists():
        print(f"Case file not found: {case_path}")
        return False
    # Open the JSON files.
    with open(case_path, 'r') as f:
        case_data = json.load(f)

    diseases_text = " ".join(case_data.get('diseases', [])).lower()
    history_text = case_data.get('patient_history', '').lower()
    symptoms_text = " ".join(case_data.get('symptoms', [])).lower()

    combined_text = diseases_text + " " + history_text + " " + symptoms_text

    if expected_disease.lower() in combined_text:
        return True

    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            return True

    return False


def calculate_precision_at_k(results, k=5):
    """Calculate average Precision@K"""
    precisions = []

    for result in results:
        # Get retrieved case IDs for THIS test case
        retrieved_cases = result['retrieved'][:k]  # Take only first K

        # Count how many are relevant
        relevant_count = 0
        for case_id in retrieved_cases:
            if is_case_relevant(
                    case_id,
                    result['expected_diagnosis'],
                    result['expected_keywords']
            ):
                relevant_count += 1

        # Calculate precision for THIS test case
        precision = relevant_count / len(retrieved_cases) if retrieved_cases else 0
        precisions.append(precision)

    # Return average across all test cases
    return sum(precisions) / len(precisions) if precisions else 0.0


if __name__ == "__main__":
    rag = ClinicalRAG()
    results = run_all_test(rag, GROUND_TRUTH)

    # Calculate metrics
    acc, correct, total = calculate_diagnosis_accuracy(results)
    avg_match = calculate_match_keywords(results)
    precision = calculate_precision_at_k(results)

    # Print to console
    print(f"\nDiagnosis Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Keyword Match: {avg_match:.2%}")
    print(f"Precision@5: {precision:.2%}")

    # NEW: Save to JSON
    metrics = {
        "diagnosis_accuracy": round(acc, 4),
        "keyword_match": round(avg_match, 4),
        "precision_at_k": round(precision, 4),
        "correct_diagnoses": correct,
        "total_cases": total,
        "timestamp": datetime.now().isoformat(),
        "model": Config.GEMINI_MODEL,
        "embedding_model": Config.TEXT_EMBEDDING_MODEL
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n Metrics saved to metrics.json")
