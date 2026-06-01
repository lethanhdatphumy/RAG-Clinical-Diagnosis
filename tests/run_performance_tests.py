#!/usr/bin/env python3
"""
Comprehensive test runner for comparing Gemma baseline performance with RAG system performance.
This script runs both baseline and RAG tests and generates a comparison report.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def ensure_results_directory():
    """Ensure the results directory exists."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def run_baseline_test():
    """Run the baseline Gemma test."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: BASELINE GEMMA TEST")
    print(f"{'=' * 80}\n")

    try:
        from test_gemma_baseline import GemmaBaselineEvaluator

        evaluator = GemmaBaselineEvaluator()
        metrics = evaluator.run_evaluation()
        evaluator.print_summary(metrics)

        # Save results
        results_dir = ensure_results_directory()
        results_file = results_dir / f"baseline_gemma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        evaluator.save_results(metrics, filepath=str(results_file))

        return metrics
    except Exception as e:
        print(f"Error running baseline test: {str(e)}")
        return None


def run_rag_test():
    """Run the RAG system test."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: RAG SYSTEM TEST")
    print(f"{'=' * 80}\n")

    try:
        from src.generation.rag_generator import ClinicalRAG
        from evaluate_rag import run_all_test, calculate_diagnosis_accuracy, calculate_match_keywords, calculate_precision_at_k
        from ground_truth import GROUND_TRUTH
        from config.settings import Config

        # Run RAG tests
        rag = ClinicalRAG()
        results = run_all_test(rag, GROUND_TRUTH)

        # Calculate metrics
        acc, correct, total = calculate_diagnosis_accuracy(results)
        avg_match = calculate_match_keywords(results)
        precision = calculate_precision_at_k(results)

        # Print results
        print(f"\n{'=' * 80}")
        print(f"RAG SYSTEM RESULTS")
        print(f"{'=' * 80}")
        print(f"Diagnosis Accuracy: {correct}/{total} = {acc:.2%}")
        print(f"Keyword Match: {avg_match:.2%}")
        print(f"Precision@5: {precision:.2%}")

        # Create metrics dict
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": Config.GEMINI_MODEL,
            "evaluation_type": "RAG System",
            "total_tests": total,
            "successful_tests": total,
            "failed_tests": 0,
            "error_rate": 0.0,
            "avg_inference_time_seconds": 0,
            "diagnosis_accuracy_percent": acc * 100,
            "avg_keyword_match_percent": avg_match * 100,
            "precision_at_5": precision * 100,
            "correct_diagnoses": correct,
            "results": results
        }

        # Save results
        results_dir = ensure_results_directory()
        results_file = results_dir / f"rag_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        def serialize_obj(obj):
            if isinstance(obj, list):
                return [serialize_obj(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_obj(v) for k, v in obj.items()}
            else:
                return str(obj)

        with open(results_file, 'w') as f:
            json.dump(serialize_obj(metrics), f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

        return metrics
    except Exception as e:
        print(f"Error running RAG test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(baseline_metrics: Dict[str, Any], rag_metrics: Dict[str, Any]):
    """Compare baseline and RAG metrics."""
    if baseline_metrics is None or rag_metrics is None:
        print("\n⚠️  Cannot compare results due to missing metrics.")
        return

    print(f"\n{'#' * 80}")
    print(f"PERFORMANCE COMPARISON: BASELINE vs RAG")
    print(f"{'#' * 80}\n")

    # Extract key metrics
    baseline_accuracy = baseline_metrics.get("diagnosis_accuracy_percent", 0)
    rag_accuracy = rag_metrics.get("diagnosis_accuracy_percent", 0)

    baseline_keyword = baseline_metrics.get("avg_keyword_match_percent", 0)
    rag_keyword = rag_metrics.get("avg_keyword_match_percent", 0)

    baseline_time = baseline_metrics.get("avg_inference_time_seconds", 0)
    rag_time = rag_metrics.get("avg_inference_time_seconds", 0)

    # Calculate improvements
    accuracy_improvement = rag_accuracy - baseline_accuracy
    accuracy_improvement_pct = (accuracy_improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0

    keyword_improvement = rag_keyword - baseline_keyword
    keyword_improvement_pct = (keyword_improvement / baseline_keyword * 100) if baseline_keyword > 0 else 0

    time_change = rag_time - baseline_time
    time_change_pct = (time_change / baseline_time * 100) if baseline_time > 0 else 0

    # Print comparison table
    print(f"{'Metric':<35} {'Baseline':<15} {'RAG':<15} {'Improvement':<15}")
    print(f"{'-' * 80}")

    print(f"{'Diagnosis Accuracy (%)':<35} {baseline_accuracy:>14.2f}% {rag_accuracy:>14.2f}% {accuracy_improvement:>+14.2f}%")
    print(f"{'Keyword Match Rate (%)':<35} {baseline_keyword:>14.2f}% {rag_keyword:>14.2f}% {keyword_improvement:>+14.2f}%")
    print(f"{'Avg Inference Time (s)':<35} {baseline_time:>14.2f}s {rag_time:>14.2f}s {time_change:>+14.2f}s")

    print(f"\n{'Percentage Improvements:':<35}")
    print(f"  Accuracy:      {accuracy_improvement_pct:>+.2f}%")
    print(f"  Keyword Match: {keyword_improvement_pct:>+.2f}%")
    print(f"  Inference Time:{time_change_pct:>+.2f}%")

    print(f"\n{'Test Statistics:':<35}")
    print(f"  Baseline Tests:     {baseline_metrics.get('total_tests', 0)}")
    print(f"  RAG Tests:          {rag_metrics.get('total_tests', 0)}")
    print(f"  Baseline Errors:    {baseline_metrics.get('error_rate', 0):.2f}%")
    print(f"  RAG Errors:         {rag_metrics.get('error_rate', 0):.2f}%")

    print(f"\n{'#' * 80}\n")


def save_comparison_report(baseline_metrics: Dict[str, Any], rag_metrics: Dict[str, Any]):
    """Save comparison report to file."""
    if baseline_metrics is None or rag_metrics is None:
        return

    results_dir = ensure_results_directory()
    report_file = results_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "model": baseline_metrics.get("model"),
            "accuracy": baseline_metrics.get("diagnosis_accuracy_percent"),
            "keyword_match": baseline_metrics.get("avg_keyword_match_percent"),
            "inference_time": baseline_metrics.get("avg_inference_time_seconds"),
            "total_tests": baseline_metrics.get("total_tests"),
            "error_rate": baseline_metrics.get("error_rate")
        },
        "rag": {
            "model": rag_metrics.get("model"),
            "accuracy": rag_metrics.get("diagnosis_accuracy_percent"),
            "keyword_match": rag_metrics.get("avg_keyword_match_percent"),
            "inference_time": rag_metrics.get("avg_inference_time_seconds"),
            "total_tests": rag_metrics.get("total_tests"),
            "error_rate": rag_metrics.get("error_rate")
        },
        "improvements": {
            "accuracy_absolute": rag_metrics.get("diagnosis_accuracy_percent", 0) - baseline_metrics.get("diagnosis_accuracy_percent", 0),
            "keyword_match_absolute": rag_metrics.get("avg_keyword_match_percent", 0) - baseline_metrics.get("avg_keyword_match_percent", 0),
            "inference_time_change": rag_metrics.get("avg_inference_time_seconds", 0) - baseline_metrics.get("avg_inference_time_seconds", 0)
        }
    }

    with open(report_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ Comparison report saved to: {report_file}")


def main():
    """Main function to run all tests."""
    print(f"\n{'#' * 80}")
    print(f"CLINICAL DIAGNOSIS SYSTEM - COMPREHENSIVE PERFORMANCE EVALUATION")
    print(f"{'#' * 80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Run baseline test
    baseline_metrics = run_baseline_test()

    # Ask if user wants to run RAG test
    print("\n" + "?" * 80)
    response = input("Would you like to run the RAG system test for comparison? (yes/no): ").strip().lower()

    rag_metrics = None
    if response in ['yes', 'y']:
        rag_metrics = run_rag_test()

        # Compare results
        if baseline_metrics and rag_metrics:
            compare_results(baseline_metrics, rag_metrics)
            save_comparison_report(baseline_metrics, rag_metrics)

    print(f"\n{'#' * 80}")
    print(f"Evaluation Complete!")
    print(f"Results Directory: tests/results/")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
