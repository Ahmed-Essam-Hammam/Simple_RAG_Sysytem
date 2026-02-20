import streamlit as st
from langchain_community.document_loaders import TextLoader
from rag_pipeline import create_vectorstore, build_rag_chain
import tempfile
import os


st.set_page_config(page_title="Arabic RAG System", layout="wide")

st.title("📚 Arabic RAG System")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name


    loader = TextLoader(temp_path, encoding="utf-8")
    documents = loader.load()


    st.success(f"Loaded {len(documents)} document(s)")

    vectorstore, docs = create_vectorstore(documents)

    st.info(f"Number of chunks created: {len(docs)}")


    rag_chain, retriever = build_rag_chain(vectorstore)

    query = st.text_input("Ask your question:")

    if query:

        # Retrieve chunks
        retrieved_docs = retriever.invoke(query)

        st.subheader("🔎 Retrieved Chunks")

        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content)

        # Get answer
        answer = rag_chain.invoke(query)

        st.subheader("🤖 Answer")
        st.write(answer)
