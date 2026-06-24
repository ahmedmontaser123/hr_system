# ai/questions_generator.py
from llm.llm_interface import LLmInterface 
from .chains import Chains

class QuestionsGenerator:
    def __init__(self, chains:Chains ):
        self.question_generator = chains.question_chain

    def generate(self, role: str, description: str) -> str:
        result = self.question_generator.invoke({
            "role": role,
            "description": description
        })

        if result is None:
            return f"As a {role} candidate, tell me about your experience with {description}."

        return result.question