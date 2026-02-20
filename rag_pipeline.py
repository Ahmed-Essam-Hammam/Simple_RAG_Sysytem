from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from embeddings import get_embeddings
from llm import get_llm
from config import CHROMA_DIR



def create_vectorstore(documents):

    embeddings = get_embeddings()


    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=100         
    # )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=85           
    )


    docs = splitter.split_documents(documents)


    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore, docs


def build_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}
    """)

    llm = get_llm()

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever