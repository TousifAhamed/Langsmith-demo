from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os

os.environ["LANGCHAIN_PROJECT"] = 'ReAct Agents'

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

# Use ChatGroq instead of ChatOpenAI
llm = ChatGroq(
    model="openai/gpt-oss-120b",  
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True  # This will handle parsing errors gracefully
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke with error handling
try:
    # Test with a simpler question first
    response = agent_executor.invoke({"input": "What is 2+2?"})
    print("=== RESPONSE ===")
    print(response)
    print("\n=== OUTPUT ===")
    print(response['output'])
except Exception as e:
    print(f"Error occurred: {e}")
    print("This might be due to rate limits or parsing issues. Try again in a moment.")