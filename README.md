# LangSmith Demo

A simple demonstration of using LangChain with Groq API for language model interactions.

## Description

This project demonstrates a basic LLM (Large Language Model) call using LangChain and Groq API. It includes a simple chain that processes questions and returns responses.

## Features

- Simple LLM chain using LangChain
- Integration with Groq API
- Environment variable management with python-dotenv
- Structured prompt templates

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

Run the simple LLM call example:
```bash
python 1_simple_llm_call.py
```

## Project Structure

```
├── 1_simple_llm_call.py    # Main script demonstrating LLM call
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (not included in repo)
├── .gitignore            # Git ignore file
└── README.md             # This file
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
