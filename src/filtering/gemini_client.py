from pathlib import Path
import time
import google.generativeai as genai
from torch.nn.utils import remove_spectral_norm
import json

from config.settings import Config


class GeminiClient:

    def __init__(self):

        genai.configure(api_key=Config.GOOGLE_API_KEY)  # Call api key from the Config class.
        self.model = genai.GenerativeModel(
            Config.GEMINI_MODEL)  # Call the Gemini model, we can replace another Google model

    def generate_response(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

    def filter_text_data(self, input_dir="../../data/processed/extracted"):
        """

        :param output_dir:
        :return:
        """
        output_path = Path(input_dir)
        """" Read all metadata.json files from extracted PDF files"""
        # Iterate over the files in this directory, and stores their names.
        case_folders = [folder for folder in output_path.iterdir() if folder.is_dir()]
        all_cases = []
        for case_folder in case_folders:
            metadata = case_folder / "metadata.json"
            if metadata:
                # Read the JSON file
                with open(metadata, 'r') as f:
                    case_data = json.load(f)

                print(f"Loaded: {case_data['pdf_name']}")
            all_cases.append(case_data)
        return all_cases

    def create_filter_prompt(self, text):
        """

        :param text:
        :return:
        """
        prompt = f"""You are a specialized biomedical NLP service. Your task is to analyze the provided medical text and extract all relevant clinical entities.

**Guidelines:**
1.  **Extract Entities:** Identify and extract terms related to the JSON keys provided.
2.  **Normalize:** All extracted string values in the lists should be **lowercase** for consistency.
3.  **Be Specific:** For measurements (vitals, labs), include the value and unit if provided (e.g., "104°f", "platelet count 70,000/µl").
4.  **Be Comprehensive:** Populate all lists with all relevant terms found in the text.
5.  **Exclude Negations:** Do NOT extract negated symptoms or conditions (e.g., "denies fever," "no history of...").
6.  **Focus on Patient:** Extract information *about the patient* in the text.
7.  **Populate Keys:**
    * `diseases`: Confirmed or suspected medical conditions (e.g., "meningitis", "malaria").
    * `symptoms`: Patient-reported complaints or observed signs (e.g., "fever", "headache", "vomiting", "lethargic").
    * `vital_signs`: Specific vital sign measurements (e.g., "fever (104°f / 40°c)").
    * `anatomical_terms`: Body parts or locations (e.g., "trunk", "extremities", "right upper lobe", "buccal mucosa").
    * `laboratory_findings`: Lab results or findings (e.g., "thrombocytopenia", "leukopenia", "eosinophilia", "positive brudzinski's sign").
    * `treatments`: Any medications or therapeutic interventions mentioned.
    * `pathogens`: Specific infectious agents (e.g., "bacterial", "meningococcal", "plasmodium").
    * `procedures`: Diagnostic tests or medical actions (e.g., "chest x-ray", "iv insertion").
    * `misc_medical_terms`: Other relevant clinical terms that don't fit above (e.g., "petechial rash", "nuchal rigidity", "paroxysmal fevers", "cavitary lesion", "lymphadenopathy").
    * `patient_history`: A *brief* summary of the patient's relevant background (e.g., "19-year-old university student", "35-year-old nurse").
    * `risk_factors`: Factors that increase risk (e.g., "living in a dormitory", "history of intravenous drug use", "returned from thailand", "swam in lake malawi").

**Text to Analyze:**
{text}

**Output Format (JSON ONLY):**
Return ONLY a valid, minified JSON object based on the schema below. Do not add any explanatory text, markdown, or apologies before or after the JSON.
{{
  "diseases": [],
  "symptoms": [],
  "vital_signs": [],
  "anatomical_terms": [],
  "laboratory_findings": [],
  "treatments": [],
  "pathogens": [],
  "procedures": [],
  "misc_medical_terms": [],
  "patient_history": "",
  "risk_factors": []
}}
"""
        return prompt

    def filter_single_page_text(self, text):

        prompt = self.create_filter_prompt(text)
        response = self.generate_response(prompt)

        # Remove Markdown code blocks
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove "```json

        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]

        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove trailing '```'

        return cleaned_response.strip()

    def process_all_cases(self):
        """

        :return: Iter all cases and filter them.
        """
        all_cases = self.filter_text_data()

        filtered_cases = []
        request_count = 0  # Count for requested times.

        for case in all_cases:
            print(f"\nProcessing case: {case['pdf_name']}")

            case_data = {
                "case_id": case['pdf_name'],
                "diseases": [],
                "symptoms": [],
                "vital_signs": [],
                "anatomical_terms": [],
                "laboratory_findings": [],
                "treatments": [],
                "pathogens": [],
                "procedures": [],
                "misc_medical_terms": [],
                "patient_history": "",
                "risk_factors": []
            }

            for page in case["pages"]:
                page_text = page["text"]

                if page_text.strip():
                    print(f"\nProcessing page number: {page['page_number']} ...")
                    if request_count > 0 and request_count % 15 == 0:
                        print("Waiting 60 seconds to avoid rate limit...")
                        time.sleep(60)

                    # Get responses filtered by LLM model.
                    filtered_response = self.filter_single_page_text(page_text)
                    request_count += 1

                    try:
                        page_data = json.loads(filtered_response)
                        case_data = self.merge_filtered_data(case_data, page_data)



                    # Manage situations where the input JSON is invalid instead of crashing the program.
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        print(filtered_response)  # See what is LLM actually returning

                    time.sleep(4.5)
            self.save_filtered_case(case_data)

            filtered_cases.append(case_data)

        return filtered_cases

    def merge_filtered_data(self, existing_data, new_page_data):
        """

        :param existing_data: accumulated data for the case
        :param new_page_data: filtered data for current page
        :return: the merged data
        """

        # List of all fields to merge
        list_fields = ["diseases", "symptoms", "vital_signs", "anatomical_terms",
                       "laboratory_findings", "treatments", "pathogens",
                       "procedures", "misc_medical_terms", "risk_factors"]

        for field in list_fields:
            # Extend the existing data by new_page_data or empty list if null.
            existing_data[field].extend(new_page_data.get(field, []))

        if new_page_data.get("patient_history", "").strip():
            if existing_data["patient_history"]:
                existing_data["patient_history"] += "\n\n" + new_page_data["patient_history"]
            else:
                existing_data["patient_history"] = new_page_data["patient_history"]

        return existing_data

    def save_filtered_case(self, case_data, output_dir="../../data/processed/filtered"):
        """

        :param output_dir: The direction of the filtered cases
        :return: Saves the output to the mentioned direction
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True,
                          exist_ok=True)
        filename = f"{case_data['case_id']}_filtered.json"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, indent=2, ensure_ascii=False)

        print(f" Saved: {filename}")


if __name__ == "__main__":
    client = GeminiClient()
    all_cases = client.process_all_cases()

    # if all_cases:
    #     # Process first-case as example
    #     first_case = all_cases[0]
    #     # Get text from 1st page
    #     if first_case['pages']:
    #         page_text = first_case['pages'][0]['text']
    #         response = client.filter_single_page_text(page_text)
    #         print(response)
    # else:
    #     print("No cases found to process")
