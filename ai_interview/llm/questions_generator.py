# ai/questions_generator.py
from llm.llm_interface import LLmInterface 
from .chains import Chains

class QuestionsGenerator:
    def __init__(self, chains:Chains ):
        self.question_generator = chains.question_chain

    def generate(self, role: str, description: str, previous_questions: list[str] | None = None) -> str:
        prev_qs = previous_questions or []
        prev_text = "\n".join(f"- {q}" for q in prev_qs) if prev_qs else "None"

        result = self.question_generator.invoke({
            "role": role,
            "description": description,
            "previous_questions": prev_text
        })

        if result is None:
            return f"As a {role} candidate, tell me about your experience with {description}."

        return result.question