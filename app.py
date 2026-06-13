import streamlit as st
import tempfile
import os
import chromadb
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="PDF Chatbot")

st.title("📄 AI PDF Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = embed_model.encode(
            texts
        ).tolist()

        client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        try:
            client.delete_collection("pdf_data")
        except:
            pass

        collection = client.create_collection(
            "pdf_data"
        )

        batch_size = 100

        for i in range(0, len(chunks), batch_size):

            docs_batch = [
                chunk.page_content
                for chunk in chunks[i:i+batch_size]
            ]

            emb_batch = embeddings[i:i+batch_size]

            ids_batch = [
                str(x)
                for x in range(i, i+len(docs_batch))
            ]

            collection.add(
                ids=ids_batch,
                documents=docs_batch,
                embeddings=emb_batch
            )

    st.success("PDF Indexed Successfully")

    question = st.text_input(
        "Ask Question"
    )

    if question:

        query_embedding = embed_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=4
        )

        context = "\n".join(
            results["documents"][0]
        )

        prompt = f"""
You are a helpful PDF assistant.

Answer only from the provided context.

If answer is not found say:
"I couldn't find that in the PDF."

Context:
{context}

Question:
{question}
"""

        with st.spinner("Thinking..."):

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role":"system",
                        "content":"Answer only from context."
                    },
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

        st.subheader("Answer")

        st.write(
            response["message"]["content"]
        )



