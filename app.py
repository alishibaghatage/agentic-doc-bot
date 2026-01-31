import streamlit as st
import os
import tempfile

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="AI Document Q&A Assistant")
st.title("📄 AI Document Q&A Assistant")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # Embedder for semantic search
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM for text generation using supported pipeline task
    llm = pipeline(
        task="text-generation",          # Changed from 'text2text-generation'
        model="google/flan-t5-base",
        device=-1,                       # CPU usage
        max_new_tokens=256               # Compatible with text-generation
    )
    return embedder, llm

embedder, llm = load_models()

# ---------------- PDF TEXT EXTRACT ----------------
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------- TEXT CHUNKING ----------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ---------------- VECTOR STORE ----------------
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    index, stored_chunks = create_faiss_index(chunks)

    st.success("PDF processed successfully ✅")

    query = st.text_input("Ask something from the document:")

    if query:
        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)

        context = " ".join([stored_chunks[i] for i in I[0]])

        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        with st.spinner("Thinking..."):
            answer = llm(prompt, do_sample=False)[0]['generated_text']

        st.write("### Answer:")
        st.write(answer)

