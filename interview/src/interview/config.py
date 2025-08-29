import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv("src/interview/.env")

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama3-8b-8192",
    temperature=0.2
)