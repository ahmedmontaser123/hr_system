from langchain_ollama import ChatOllama
from llm.llm_interface import LLmInterface
from helpers import Settings

class OllamaProvider(LLmInterface):
    def __init__(self, settings: Settings):
        self.model_name = settings.OLLAMA_MODEL
        self.llm = self._load()

    def _load(self):
        return ChatOllama(
            model=self.model_name,
            temperature=0.6,
            num_predict=10,
        )

    def get_llm(self):
        return self.llm