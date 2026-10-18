from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from config.settings import Config


class ClinicalRAG:
    def __init__(self):
        print("Connecting to Qdrant...")
        self.vector_store = self._load_qdrant_store()

        print("Initialising Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=512,
        )

        self.prompt = self._create_prompt()

        print("Creating RAG chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": Config.TOP_K_RETRIEVAL}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )
        print("RAG system ready!")

    def _load_qdrant_store(self) -> Qdrant:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.TEXT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )
        return Qdrant(
            client=client,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            embeddings=embeddings,
            content_payload_key="page_content",
        )

    def _create_prompt(self) -> PromptTemplate:
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

    def query(self, patient_symptoms: str) -> dict:
        """Query the RAG system for diagnosis."""
        print(f"Query: {patient_symptoms}\n")
        print("Searching Qdrant for similar cases...")
        return self.qa_chain.invoke({"query": patient_symptoms})


if __name__ == "__main__":
    rag = ClinicalRAG()

    test_query = "Patient with high fever, bleeding, and recent travel to West Africa. What could be the diagnosis?"
    result = rag.query(test_query)

    print("=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    print(result["result"])

    print("\n" + "=" * 60)
    print("SIMILAR CASES USED:")
    print("=" * 60)
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n{i}. {doc.metadata.get('case_id', f'Case {i}')}")
        print(f"   {doc.page_content[:150]}...")
