"""
Performance test for the base Gemma model before RAG integration.
This test evaluates the model's ability to diagnose tropical diseases without retrieval augmentation.
"""
import time
import json
from typing import Dict, List, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config.settings import Config

# Test cases with expected diagnoses
GEMMA_BASELINE_TESTS = [
    {
        "query": "20-year-old Sudanese refugee in Uganda with fever, bleeding from gums, vomiting, diarrhea, and subconjunctival hemorrhage during outbreak",
        "expected_diagnosis": "ebola",
        "expected_keywords": ["ebola", "hemorrhagic", "filovirus", "viral HF"]
    },
    {
        "query": "45-year-old farmer from Bangladesh with prolonged fever, massive splenomegaly, pancytopenia, and darkened skin after sandfly bites",
        "expected_diagnosis": "visceral leishmaniasis",
        "expected_keywords": ["leishmania", "kala-azar", "sandfly", "splenomegaly"]
    },
    {
        "query": "38-year-old from Philippines with fever, jaundice, calf pain, and renal failure after wading through floodwater with rats",
        "expected_diagnosis": "leptospirosis",
        "expected_keywords": ["leptospira", "weil's disease", "jaundice", "renal failure"]
    },
    {
        "query": "28-year-old pregnant woman from Brazil with mild fever, rash, conjunctivitis, and arthralgia after mosquito bites",
        "expected_diagnosis": "zika virus infection",
        "expected_keywords": ["zika", "flavivirus", "microcephaly", "aedes mosquito"]
    },
    {
        "query": "32-year-old safari guide from Tanzania with cyclical fever every 48 hours, chills, sweating, and splenomegaly",
        "expected_diagnosis": "malaria",
        "expected_keywords": ["malaria", "plasmodium", "paroxysmal fever", "splenomegaly"]
    },
    {
        "query": "50-year-old homeless man with chronic cough, hemoptysis, night sweats, weight loss, and cavitary lung lesion",
        "expected_diagnosis": "tuberculosis",
        "expected_keywords": ["tuberculosis", "mycobacterium", "cavitary", "pulmonary"]
    },
    {
        "query": "24-year-old student in dormitory with sudden fever, severe headache, neck stiffness, photophobia, and petechial rash",
        "expected_diagnosis": "meningitis",
        "expected_keywords": ["meningitis", "meningococcal", "petechial", "nuchal rigidity"]
    },
    {
        "query": "36-year-old returning from Caribbean with high fever and severe debilitating joint pain in wrists and ankles after daytime mosquito bites",
        "expected_diagnosis": "chikungunya",
        "expected_keywords": ["chikungunya", "alphavirus", "arthralgia", "aedes"]
    },
    {
        "query": "42-year-old from Amazon rainforest with jaundice, hemorrhage, coffee-ground vomiting, and slow pulse despite high fever, unvaccinated",
        "expected_diagnosis": "yellow fever",
        "expected_keywords": ["yellow fever", "flavivirus", "jaundice", "hemorrhagic"]
    },
    {
        "query": "31-year-old from South Asia with stepladder fever, rose spots, relative bradycardia, and hepatosplenomegaly after eating street food",
        "expected_diagnosis": "typhoid fever",
        "expected_keywords": ["typhoid", "salmonella typhi", "enteric fever", "rose spots"]
    },
    {
        "query": "29-year-old IV drug user with weight loss, oral thrush, lymphadenopathy, recurrent herpes zoster, and low CD4 count",
        "expected_diagnosis": "hiv",
        "expected_keywords": ["hiv", "aids", "immunodeficiency", "opportunistic infection"]
    },
    {
        "query": "27-year-old from West Africa with painless non-healing ulcer on forearm with raised border after insect bites",
        "expected_diagnosis": "cutaneous leishmaniasis",
        "expected_keywords": ["leishmaniasis", "cutaneous", "sandfly", "ulcer"]
    },
    {
        "query": "30-year-old returning from Thailand with high fever, retro-orbital pain, rash, bleeding gums, and low platelets",
        "expected_diagnosis": "dengue fever",
        "expected_keywords": ["dengue", "hemorrhagic", "thrombocytopenia", "aedes"]
    },
    {
        "query": "26-year-old after swimming in Lake Kariba with bloody diarrhea, hepatosplenomegaly, hematuria, and high eosinophils",
        "expected_diagnosis": "schistosomiasis",
        "expected_keywords": ["schistosomiasis", "bilharzia", "freshwater", "eosinophilia"]
    },
    {
        "query": "44-year-old on steroids from Cambodia with diarrhea, wheezing, moving skin rash, extremely high eosinophils, and larvae in stool",
        "expected_diagnosis": "strongyloidiasis",
        "expected_keywords": ["strongyloides", "hyperinfection", "larva currens", "eosinophilia"]
    }
]


class GemmaBaselineEvaluator:
    """Evaluates the base Gemma model performance without RAG."""

    def __init__(self):
        print(f"Initializing Gemma Baseline Evaluator...")
        print(f"Model: {Config.GEMINI_MODEL}")

        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=512
        )

        self.prompt = self.create_prompt()
        self.results = []

    def create_prompt(self) -> PromptTemplate:
        """Create a medical diagnosis prompt for baseline evaluation."""
        template = """You are an expert tropical medicine physician with extensive knowledge of infectious diseases.

        Patient Query: {question}
        
        Based on your medical knowledge, provide:
        1. Most likely diagnosis (be specific with the disease name)
        2. Key supporting clinical evidence from the patient presentation
        3. Differential diagnoses to consider
        4. Recommended diagnostic tests
        5. Suggested treatment approach
        
        Answer:"""

        return PromptTemplate(template=template, input_variables=["question"])

    def extract_diagnosis(self, response: str) -> str:
        """Extract the primary diagnosis from the model response."""
        response_lower = response.lower()

        # Try to find diagnosis in common patterns
        lines = response_lower.split('\n')
        for line in lines:
            if 'diagnosis' in line or 'likely' in line:
                # Extract text after colon or the line itself
                if ':' in line:
                    diagnosis = line.split(':', 1)[1].strip()
                else:
                    diagnosis = line.replace('diagnosis', '').replace('likely', '').strip()

                # Clean up the diagnosis
                diagnosis = diagnosis.split('\n')[0].strip('- ')
                if diagnosis:
                    return diagnosis

        return ""

    def calculate_keyword_match(self, response: str, expected_keywords: List[str]) -> float:
        """Calculate the percentage of expected keywords found in the response."""
        if not expected_keywords:
            return 0.0

        response_lower = response.lower()
        matches = 0

        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                matches += 1

        return (matches / len(expected_keywords)) * 100

    def test_single_query(self, test_case: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """Test a single query and return metrics."""
        query = test_case["query"]
        expected_diagnosis = test_case["expected_diagnosis"].lower()
        expected_keywords = test_case["expected_keywords"]

        print(f"\n{'=' * 80}")
        print(f"TEST {test_index + 1}/{len(GEMMA_BASELINE_TESTS)}")
        print(f"{'=' * 80}")
        print(f"\nQuery: {query}\n")
        print(f"Expected Diagnosis: {expected_diagnosis}")

        # Measure response time
        start_time = time.time()
        try:
            response = self.llm.invoke(self.prompt.format(question=query))
            response_text = response.content
            inference_time = time.time() - start_time
        except Exception as e:
            print(f"Error querying model: {str(e)}")
            return {
                "test_index": test_index,
                "query": query,
                "expected_diagnosis": expected_diagnosis,
                "error": str(e),
                "inference_time": None,
                "response": None,
                "extracted_diagnosis": None,
                "keyword_match": 0.0,
                "diagnosis_correct": False
            }

        # Extract diagnosis
        extracted_diagnosis = self.extract_diagnosis(response_text)

        # Calculate metrics
        keyword_match = self.calculate_keyword_match(response_text, expected_keywords)
        diagnosis_correct = expected_diagnosis in extracted_diagnosis.lower()

        print(f"\nModel Response:")
        print(f"{response_text}")
        print(f"\nExtracted Diagnosis: {extracted_diagnosis}")
        print(f"Inference Time: {inference_time:.2f}s")
        print(f"Keyword Match: {keyword_match:.1f}%")
        print(f"Diagnosis Correct: {diagnosis_correct}")

        result = {
            "test_index": test_index,
            "query": query,
            "expected_diagnosis": expected_diagnosis,
            "response": response_text,
            "extracted_diagnosis": extracted_diagnosis,
            "inference_time": inference_time,
            "keyword_match": keyword_match,
            "diagnosis_correct": diagnosis_correct,
            "expected_keywords": expected_keywords,
            "error": None
        }

        return result

    def run_evaluation(self) -> Dict[str, Any]:
        """Run the full baseline evaluation."""
        print(f"\n{'#' * 80}")
        print(f"GEMMA BASELINE EVALUATION - WITHOUT RAG")
        print(f"{'#' * 80}")
        print(f"Model: {Config.GEMINI_MODEL}")
        print(f"Total Test Cases: {len(GEMMA_BASELINE_TESTS)}")
        print(f"Evaluation Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.results = []

        for i, test_case in enumerate(GEMMA_BASELINE_TESTS):
            result = self.test_single_query(test_case, i)
            self.results.append(result)

        # Calculate aggregate metrics
        metrics = self.calculate_metrics()

        return metrics

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.get("error") is None)

        if successful_tests == 0:
            return {
                "total_tests": total_tests,
                "successful_tests": 0,
                "error_rate": 100.0,
                "avg_inference_time": 0,
                "keyword_match_rate": 0.0,
                "diagnosis_accuracy": 0.0,
                "results": self.results
            }

        valid_results = [r for r in self.results if r.get("error") is None]

        keyword_matches = [r["keyword_match"] for r in valid_results]
        inference_times = [r["inference_time"] for r in valid_results if r["inference_time"] is not None]
        diagnosis_correct_count = sum(1 for r in valid_results if r["diagnosis_correct"])

        avg_keyword_match = sum(keyword_matches) / len(keyword_matches) if keyword_matches else 0.0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        diagnosis_accuracy = (diagnosis_correct_count / len(valid_results)) * 100

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": Config.GEMINI_MODEL,
            "evaluation_type": "Baseline (Without RAG)",
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "error_rate": ((total_tests - successful_tests) / total_tests) * 100,
            "avg_inference_time_seconds": avg_inference_time,
            "avg_keyword_match_percent": avg_keyword_match,
            "diagnosis_accuracy_percent": diagnosis_accuracy,
            "results": self.results
        }

        return metrics

    def print_summary(self, metrics: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        print(f"\n{'#' * 80}")
        print(f"EVALUATION SUMMARY - GEMMA BASELINE")
        print(f"{'#' * 80}")
        print(f"\nTimestamp: {metrics['timestamp']}")
        print(f"Model: {metrics['model']}")
        print(f"Evaluation Type: {metrics['evaluation_type']}")
        print(f"\n{'─' * 80}")
        print(f"Test Results:")
        print(f"  Total Tests:        {metrics['total_tests']}")
        print(f"  Successful Tests:   {metrics['successful_tests']}")
        print(f"  Failed Tests:       {metrics['failed_tests']}")
        print(f"  Error Rate:         {metrics['error_rate']:.2f}%")
        print(f"\n{'─' * 80}")
        print(f"Performance Metrics:")
        print(f"  Avg Inference Time: {metrics['avg_inference_time_seconds']:.2f}s")
        print(f"  Keyword Match Rate: {metrics['avg_keyword_match_percent']:.2f}%")
        print(f"  Diagnosis Accuracy: {metrics['diagnosis_accuracy_percent']:.2f}%")
        print(f"\n{'#' * 80}\n")

    def save_results(self, metrics: Dict[str, Any], filepath: str = None):
        """Save evaluation results to a JSON file."""
        if filepath is None:
            filepath = f"baseline_gemma_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert non-serializable objects to strings
        def serialize_obj(obj):
            if isinstance(obj, list):
                return [serialize_obj(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_obj(v) for k, v in obj.items()}
            else:
                return str(obj)

        serializable_metrics = serialize_obj(metrics)

        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    evaluator = GemmaBaselineEvaluator()
    metrics = evaluator.run_evaluation()
    evaluator.print_summary(metrics)
    evaluator.save_results(metrics, filepath="tests/results/baseline_gemma_results.json")
