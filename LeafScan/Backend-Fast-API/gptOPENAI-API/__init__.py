# app/llm/__init__.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get GPT API key from environment
GPT_TOKEN = os.getenv("GPT_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=GPT_TOKEN)
