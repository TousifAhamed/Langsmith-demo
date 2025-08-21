import os
from pathlib import Path 
from dotenv import load_dotenv
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

loader = PyPDFLoader(pdf_path)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# HuggingFace Embeddings (Free alternative - no API key required)
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Correct embedding model
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Alternative models you can try:
# emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Better quality, slower
# emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # Multilingual support

# Create vector store  
vector_store = FAISS.from_documents(chunks, emb)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

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

rag_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | StrOutputParser()
)

# Example usage
if __name__ == "__main__":
    question = input("\nQ: ")
    result = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {result}")