import sys
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logging_utils import logging

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")


def openai_client(model_name='gpt-4o-mini', max_tokens=10000):
    return ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=api_key, max_tokens=max_tokens)


def init_client(prompt, model_pro='gpt', model_type='gpt-4o-mini', max_tokens=10000, stream=False):
    if model_pro == 'gpt':
        op_client = OpenAI(api_key=api_key)
        response = op_client.chat.completions.create(
            model=model_type,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=stream
        )
        if stream:
            return response
        return response.choices[0].message.content.strip()
    if model_pro == 'groq':
        groq_client = Groq(api_key=groq_api_key)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=stream
        )
        if stream:
            return response
        return response.choices[0].message.content.strip()
    if model_pro == 'hf':
        from huggingface_hub import InferenceClient
        hf_client = InferenceClient(provider="nebius", api_key=hf_api_key)
        response = hf_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=0.7,
        )
        return response.choices[0].message.content.strip()


def groq_client():
    return Groq(api_key=groq_api_key)
