import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_PROJECT"] = 'SequentialChain LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}?",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 point summary of {text}.",
    input_variables=["text"]
)

model1 = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# qwen/qwen3-32b

model2 = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)
parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {
        'model1': 'openai/gpt-oss-120b',
        'model1_temp': 0.7,
        'parser': 'stroutputparser',
        'model2': 'qwen/qwen3-32b',
        'model2_temp': 0.5
        
    }
}

result = chain.invoke({"topic": "Artificial Intelligence in Karnataka"}, config=config)

print(result)
