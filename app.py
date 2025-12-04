import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(page_title="AI Document Q&A Agent")
st.title("📄 AI Document Q&A Assistant")

# -------------------------
# PDF uploader
# -------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file_path = "temp_uploaded.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # -------------------------
    # Load and split PDF
    # -------------------------
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # -------------------------
    # Create embeddings and vector store
    # -------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)

    # -------------------------
    # HuggingFace LLM pipeline
    # -------------------------
    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # -------------------------
    # Retrieval QA chain
    # -------------------------
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

    # -------------------------
    # User question input
    # -------------------------
    query = st.text_input("Ask something from the document:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.write("### Answer:")
            st.write(answer)



