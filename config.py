import os
from dotenv import load_dotenv

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "gpt-oss-120b"