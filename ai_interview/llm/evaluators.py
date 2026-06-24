# ai/evaluators.py
from .chains import Chains
from llm.llm_interface import LLmInterface

class Evaluator:
    def __init__(self, chains:Chains ):
        self.evaluator = chains.evaluation_chain
    

    def evaluate_answer(self, question: str, answer: str, category: str):
        result = self.evaluator.invoke({
            "question": question,
            "answer": answer,
            "category": category
        })

        if result is None:
            return {
                "status": "evaluated",
                "evaluation": {
                    "score": 5,
                    "feedback": "Your answer was received but could not be fully evaluated. Consider providing more specific details and examples in your response."
                }
            }

        return {
            "status": "evaluated",
            "evaluation": {
                "score": result.score,
                "feedback": result.feedback
            }
        }

      