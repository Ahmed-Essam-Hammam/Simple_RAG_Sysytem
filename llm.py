from langchain_openai import ChatOpenAI
from config import CEREBRAS_API_KEY, LLM_MODEL

def get_llm():
    return ChatOpenAI(
        base_url="https://api.cerebras.ai/v1",
        model=LLM_MODEL,
        api_key=CEREBRAS_API_KEY,
        temperature=0
    )
