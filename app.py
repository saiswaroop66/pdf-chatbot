import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import tempfile
import ollama

st.title("📄 PDF Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts).tolist()

    # ChromaDB
    client = chromadb.Client()

    collection = client.get_or_create_collection("pdf_data")

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk.page_content],
            embeddings=[embeddings[i]]
        )

    st.success("PDF Loaded Successfully!")

    question = st.text_input("Ask a Question")

    if question:

        query_embedding = model.encode([question]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2
        )

        context = "\n".join(results["documents"][0])

        prompt = f"""
Answer only from the given context.

Context:
{context}

Question:
{question}
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.subheader("Answer")
        st.write(response["message"]["content"])


