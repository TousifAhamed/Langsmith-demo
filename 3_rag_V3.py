import os
from pathlib import Path 
from dotenv import load_dotenv
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# Using HuggingFace embeddings (free alternative)
# from langchain_openai import OpenAIEmbeddings  # Option 1: OpenAI (paid, high quality)
from langchain_huggingface import HuggingFaceEmbeddings  # Option 2: HuggingFace (free)

os.environ["LANGCHAIN_PROJECT"] = 'RAG Chatbot'

load_dotenv()

pdf_path = Path("Vision AI .pdf")

@traceable(name="pdf_loader")
def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Correct embedding model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(splits, emb)
    return vector_store

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vector_store = build_vectorstore(splits)
    return vector_store


# Initialize Groq model
model = ChatGroq(
    model="openai/gpt-oss-120b",  # or your preferred Groq model
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer: """)

# Create the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# setup pipeline and query docs
@traceable(name="setup_pipeline_and_query_docs")
def setup_pipeline_and_query_docs(pdf_path: str, question: str):
    # Setup the pipeline
    vector_store = setup_pipeline(pdf_path)
    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


    rag_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    result = rag_chain.invoke(question, config=config)
    
    return result
# Example usage
if __name__ == "__main__":
    config = {
        "run_name": "pdf_rag_full_query"
    }
    question = input("\nQ: ").strip()
    result = setup_pipeline_and_query_docs(pdf_path, question)
    print(f"Question: {question}")
    print(f"Answer: {result}")