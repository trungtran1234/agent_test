from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

app = FastAPI()

load_dotenv()

#pydantic
class QueryRequest(BaseModel):
    query: str

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0)

def get_weather(location: str) -> str:
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "imperial"
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        return f"Cant find {location} bruh"
    
    data = response.json()
    weather_desc = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    
    weather_info = (
        f"The current weather in {location} is {weather_desc} with a temperature of {temperature}°F "
    )
    return weather_info

@tool("weather_tool", return_direct=True)
def tool_get_weather(query: str) -> str:
    """ Get the weather of a location """
    return get_weather(query)

tools = [
    tool_get_weather,
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@app.post("/agent")
async def agent_endpoint(request: QueryRequest):
    query = request.query
    try:
        response = agent.run(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
