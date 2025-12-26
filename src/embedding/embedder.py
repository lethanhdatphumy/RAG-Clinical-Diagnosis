from pathlib import Path
import json
import os
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Get the absolute path to THIS script file
THIS_FILE = os.path.abspath(__file__)
# 2. Get the directory this script is in
SCRIPT_DIR = os.path.dirname(THIS_FILE)
# 3. Get the project root (go up two levels)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 4. Build the data path from the ROOT using os.path.join
FILTERED_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "filtered"
)
# 5. Build the data path from the ROOT to vector_store
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "vector", "clinical_faiss")


class ClinicalEmbedder:

    def __init__(self):
        self.filtered_dir = Path(FILTERED_DATA_PATH)

    def load_filtered_cases(self):
        """
        :return: Load all filtered JSON cases.
        """
        json_files = [file for file in self.filtered_dir.iterdir() if file.suffix == '.json']
        # print(json_files[:5])
        print(f"Found: {len(json_files)} filtered files")

        cases = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                case_data = json.load(f)
                cases.append(case_data)
        return cases

    def prepare_text_for_embedding(self, case_data):
        """
        :param case_data: data of JSON filtered case.
        :return: combine text string.
        """

        text_parts = []  # Combine all the important fields

        # Add patient history
        if case_data.get("patient_history"):
            text_parts.append(f"Patient History: {case_data['patient_history']}")

        # Add diseases
        if case_data.get("diseases"):
            diseases = ', '.join(case_data["diseases"])  # Join the list into a string
            text_parts.append(f"Diseases: {diseases}")

        # Add symptoms
        if case_data.get("symptoms"):
            symptoms = ', '.join(case_data["symptoms"])
            text_parts.append(f"Symptoms: {symptoms}")

        # Add treatments
        if case_data.get("treatments"):
            treatments = ', '.join(case_data["treatments"])
            text_parts.append(f"Treatments: {treatments}")

        # Add laboratory findings
        if case_data.get("laboratory_findings"):
            lab_findings = ', '.join(case_data["laboratory_findings"])
            text_parts.append(f"Laboratory Findings: {lab_findings}")

        # Add treatments
        if case_data.get("risk_factors"):
            risk_factors = ', '.join(case_data["risk_factors"])
            text_parts.append(f"Risk Factors: {risk_factors}")

        # Add pathogens
        if case_data.get("pathogens"):
            pathogens = ', '.join(case_data["pathogens"])
            text_parts.append(f"Pathogens: {pathogens}")

        # Add procedures
        if case_data.get("procedures"):
            procedures = ', '.join(case_data["procedures"])
            text_parts.append(f"Procedures: {procedures}")

        # Add vital signs
        if case_data.get("vital_signs"):
            vital_signs = ', '.join(case_data["vital_signs"])
            text_parts.append(f"Vital Signs: {vital_signs}")

        # Combine all parts into a single string
        combined_text = "\n".join(text_parts)
        return combined_text

    def load_filtered_as_document(self):
        """
        :return: load cases data as Langchain Document.
        """
        documents = []
        # Load the filtered data.
        cases = self.load_filtered_cases()

        # For each case, create a document.
        for case in cases:
            text = self.prepare_text_for_embedding(case)

            # Create Langchain Document.
            doc = Document(
                page_content=text,
                metadata={
                    "case_id": case["case_id"],
                }
            )
            documents.append(doc)

        print(f"Created {len(documents)}")
        return documents

    def create_vector_store(self, documents, model_name="all-MiniLM-L6-v2"):
        """
        :param documents: Langchain documents,
        :param model_name: HuggingFace embedding model name
        :return: FAISS vector store
        """
        print(f"\n{'=' * 60}")
        print(f"Creating embeddings using: {model_name}")
        print(f"{'=' * 60}")

        # Embedding
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}  # use CPU instead of cuda
        )

        # Create FAISS vector store from documents.
        print(f"Embedding {len(documents)} documents...")
        vector_store = FAISS.from_documents(documents, embeddings)

        print(f"Vector store created with {len(documents)} vectors")

        return vector_store

    def save_vector_store(self, vector_store, save_path=VECTOR_STORE_PATH):
        """
        :param vector_store: vectorized data
        :param sava_path: the direction for saving vector_store cases
        :return: Saves the output to the mentioned direction.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True,
                        exist_ok=True)

        vector_store.save_local(save_path)
        print(f" Vector store saved to: {save_path}")

    def load_vector_store(self, load_path=VECTOR_STORE_PATH, model_name="all-MiniLM-L6-v2"):
        """
        :param model_name: The model used to embed
        :param load_path: The direction of the local disk
        :return: Load the existing FAISS vector from the disk.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )

        vector_store = FAISS.load_local(load_path, embeddings)
        print(f" Vector store loaded from: {load_path}")
        return vector_store


if __name__ == "__main__":
    embedder = ClinicalEmbedder()

    # Load documents
    documents = embedder.load_filtered_as_document()

    # Create vector store
    vector_store = embedder.create_vector_store(documents)

    # Save it!
    embedder.save_vector_store(vector_store)

    # Test search
    print(f"\n{'=' * 60}")
    print("Testing similarity search...")
    print(f"{'=' * 60}")
    results = vector_store.similarity_search("patient with fever and malaria", k=3)

    print(f"\nTop 3 similar cases:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. Case ID: {doc.metadata['case_id']}")
        print(f"   Preview: {doc.page_content[:150]}...")

