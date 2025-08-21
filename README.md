# LangSmith Demo

A comprehensive demonstration of using LangChain with Groq API for various AI applications including simple LLM calls, sequential chains, and RAG (Retrieval-Augmented Generation) implementations.

## Description

This project showcases multiple LangChain implementations:
- Basic LLM interactions with Groq API
- Sequential chain processing
- RAG (Retrieval-Augmented Generation) with PDF documents
- Vector embeddings using HuggingFace models
- Integration with LangSmith for tracing and monitoring

## Features

- **Simple LLM Chain**: Basic question-answering using Groq API
- **Sequential Chains**: Multi-step processing workflows
- **RAG Implementation**: Document-based question answering with PDF support
- **Free Embeddings**: Using HuggingFace sentence-transformers (no API key required)
- **Vector Storage**: FAISS for efficient similarity search
- **LangSmith Integration**: Tracing and monitoring capabilities
- **Environment Management**: Secure API key handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TousifAhamed/Langsmith-demo.git
cd Langsmith-demo
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the root directory and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### 1. Simple LLM Call
```bash
python 1_simple_llm_call.py
```

### 2. Sequential Chain Processing
```bash
python 2_sequential_chain.py
```

### 3. RAG (Retrieval-Augmented Generation)
```bash
# Basic RAG implementation
python 3_rag_v1.py

# Enhanced RAG version
python 3_rag_v2.py

# Latest RAG with LangSmith tracing
python 3_rag_V3.py
```

## Project Structure

```
├── 1_simple_llm_call.py    # Basic LLM interaction with Groq
├── 2_sequential_chain.py   # Sequential chain processing
├── 3_rag_v1.py             # Basic RAG with HuggingFace embeddings
├── 3_rag_v2.py             # Enhanced RAG implementation  
├── 3_rag_V3.py             # Latest RAG with LangSmith tracing
├── requirements.txt        # Python dependencies
├── Vision AI .pdf          # Sample PDF for RAG testing
├── .env                    # Environment variables (not included in repo)
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Requirements

- Python 3.7+
- Groq API key
- Internet connection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
