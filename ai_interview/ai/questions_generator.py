
class QuestionGenerator:
    def __init__(self, chains: dict):
        self.chains = chains

    def generate(
        self,
        role: str,
        description: str,
        topic: str,
        difficulty: str,
        previous_topics: list,
    ) -> str:
        return self.chains["question_generation"].invoke({
            "role": role,
            "description": description,
            "topic": topic,
            "difficulty": difficulty,
            "previous_topics": ", ".join(previous_topics) if previous_topics else "None",
        })

    def decide_next_step(
        self,
        role: str,
        question: str,
        answer: str,
        evaluation: dict,
        previous_topics: list,
    ) -> dict:
        return self.chains["next_question_decision"].invoke({
            "role": role,
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "previous_topics": ", ".join(previous_topics) if previous_topics else "None",
        })