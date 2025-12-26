from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config.settings import Config

from src.embedding.embedder import ClinicalEmbedder


class ClinicalRAG:
    def __init__(self):
        print("Loading vector store ...")
        self.embedder = ClinicalEmbedder()
        self.vector_store = self.embedder.load_vector_store()

        print("Initialize Gemini ...")
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=512  # Creativity dial for the AI. Control the randomness of output
        )

        self.prompt = self.create_prompt()

        print("Creating RAG chain ...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Method for handling documents
            retriever=self.vector_store.as_retriever(search_kwargs={"k": Config.TOP_K_RETRIEVAL}),
            # Tell the chain where get its knowledge, and retrieve the top k most relevant

            return_source_documents=True,  # return the actual k documents that retrieved from database.
            chain_type_kwargs={"prompt": self.prompt}
            # By default, RetrievalQA uses a generic prompt. This customizes the prompt sent to the LLM.
        )
        print("RAG system ready !")

    def create_prompt(self):
        """Create medical diagnosis prompt"""

        template = """You are an expert tropical medicine physician. 
            Use the following case reports to help diagnose the patient's condition.
        
            Context from similar cases:
            {context}
        
            Patient Query: {question}
        
            Based on the similar cases above, provide:
            1. Most likely diagnosis
            2. Key supporting evidence from the cases
            3. Recommended diagnostic tests
            4. Suggested treatment approach
        
            Answer:"""

        return PromptTemplate(template=template, input_variables=["context", "question"])
        # Constructs and returns a PromptTemplate.

    def query(self, patient_symptoms):
        """Query the RAG system for diagnosis"""
        print(f"Query: {patient_symptoms}\n")
        print("Searching for similar cases...")

        result = self.qa_chain({"query": patient_symptoms})

        return result


if __name__ == "__main__":
    rag = ClinicalRAG()

    # Test query
    test_query = "Patient with high fever, bleeding, and recent travel to West Africa. What could be the diagnosis?"

    result = rag.query(test_query)

    print("=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    print(result['result'])

    print("\n" + "=" * 60)
    print("SIMILAR CASES USED:")
    print("=" * 60)
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n{i}. {doc.metadata['case_id']}")
        print(f"   {doc.page_content}...")
