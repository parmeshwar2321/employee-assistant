from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature,
        google_api_key=settings.google_api_key or None,
    )
