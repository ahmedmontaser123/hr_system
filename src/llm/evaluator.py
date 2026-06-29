# ai/evaluators.py
from .chains import Chains

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
                "score":0,
                "feedback": "there is a problem or issue"
            }
        
        return {

            "score": result.score,
            "feedback": result.feedback
        }




        