# ai/evaluators.py
from llm import build_chains
from llm.llm_interface import LLmInterface

class Evaluator:
    def __init__(self, chains:dict ):
        self.chains = chains

    def check_relevance(self, question: str, answer: str) -> dict:
        return self.chains["relevance"].invoke({"question": question, "answer": answer})

    def classify(self, question: str) -> dict:
        return self.chains["classification"].invoke({"question": question})

    def route(self, question: str, answer: str, question_type: str) -> dict:
        payload = {"question": question, "answer": answer}

        if question_type == "technical":
            return self.chains["technical"].invoke(payload)
        elif question_type == "problem_solving":
            return self.chains["problem"].invoke(payload)
        else:
            return self.chains["behavioral"].invoke(payload)

    def process(self, question: str, answer: str) -> dict:
        relevance = self.check_relevance(question, answer)

        if relevance.get("score", 0) < 2:
            return {"status": "not_relevant", "relevance": relevance}

        classification = self.classify(question)
        question_type = classification.get("type", "behavioral")

        result = self.route(question, answer, question_type)

        return {
            "status": "evaluated",
            "question_type": question_type,
            "relevance": relevance,
            "evaluation": result,
        }