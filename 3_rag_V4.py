import os
import json
import hashlib
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


load_dotenv()

pdf_path = Path("Vision AI .pdf")
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

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
def build_vectorstore(splits, embedding_model_name):
    emb = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(splits, emb)
    return vector_store

def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}

def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embedding_model: str) -> str:
    meta ={
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "format": "v1"
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path, embedding_model: str):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Correct embedding model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
    return vector_store

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = load_pdf(pdf_path)  # child
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # child
    vs = build_vectorstore(splits, embed_model_name)  # child
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

# ----------------- dispatcher (not traced) -----------------
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

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

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", force_rebuild: bool = False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(pdf_path, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | model | StrOutputParser()

    return chain.invoke(
        question,
        config={"run_name": "pdf_rag_query", "tags": ["qa"], "metadata": {"k": 4}}
    )


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
        "run_name": "pdf_rag_full_run_query"
    }
    question = input("\nQ: ").strip()
    result = setup_pipeline_and_query_docs(pdf_path, question)
    print(f"Question: {question}")
    print(f"Answer: {result}")