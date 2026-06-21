# ai/questions_generator.py
from llm.llm_interface import LLmInterface 
from llm import build_chains

class QuestionsGenerator:
    def __init__(self, llm_provider: LLmInterface ):
        self.chains = build_chains(llm_provider)

    def generate(self, role: str, description: str) -> str:
        return self.chains["question_generation"].invoke({
            "role": role,
            "description": description
        })