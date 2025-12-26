import time

import streamlit as st
from src.generation.rag_generator import ClinicalRAG

st.set_page_config(
    page_title="Clinical Diagnosis Assistant",
    layout="wide"
)

st.title("Clinical Diagnosis Assistant")
st.markdown("AI-powered diagnostic support for tropical and infectious diseases")


# Initialize RAG system
@st.cache_resource
def load_rag_generator():
    rag = ClinicalRAG()
    return rag


# Create text area for symptom input
greet = st.text_input("Tell me about your health.")

# Create a button for generating response
button = st.button("Get response!")

if button:
    st.write("Starting process!")

    with st.spinner("Wait for it ..."):
        rag = load_rag_generator()
        response = rag.query(greet)
        st.write(response['result'])
