from enum import Enum

class LLMProviderType(Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GROQ = "groq"

class HuggingFaceModels(Enum):
    QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
