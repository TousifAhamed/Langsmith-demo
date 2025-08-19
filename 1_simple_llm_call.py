import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = PromptTemplate.from_template("{question}")
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of India?"})
print(result)

