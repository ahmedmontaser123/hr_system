from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class QARecord:
    question: str
    answer: str
    topic: str
    difficulty: str
    evaluation: Dict
    relevance: Dict
    question_type: str


@dataclass
class InterviewSession:
    candidate_name: str
    role: str
    description: str

    current_topic: Optional[str] = None
    current_difficulty: str = "Easy"

    previous_topics: List[str] = field(default_factory=list)
    history: List[QARecord] = field(default_factory=list)

    def add_record(
        self,
        question: str,
        answer: str,
        topic: str,
        difficulty: str,
        evaluation: Dict,
        relevance: Dict,
        question_type: str,
    ):
        record = QARecord(
            question=question,
            answer=answer,
            topic=topic,
            difficulty=difficulty,
            evaluation=evaluation,
            relevance=relevance,
            question_type=question_type,
        )
        self.history.append(record)

        if topic not in self.previous_topics:
            self.previous_topics.append(topic)

    def update_next_step(self, next_topic: str, difficulty: str):
        self.current_topic = next_topic
        self.current_difficulty = difficulty

    def get_average_score(self, key: str = "score") -> float:
        """Average a numeric field across all evaluations (e.g. 'correctness', 'score')."""
        scores = [
            r.evaluation.get(key)
            for r in self.history
            if isinstance(r.evaluation.get(key), (int, float))
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def get_covered_topics(self) -> List[str]:
        return self.previous_topics

    def get_summary(self) -> Dict:
        return {
            "candidate": self.candidate_name,
            "role": self.role,
            "topics_covered": self.previous_topics,
            "total_questions": len(self.history),
            "history": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "topic": r.topic,
                    "difficulty": r.difficulty,
                    "type": r.question_type,
                    "evaluation": r.evaluation,
                }
                for r in self.history
            ],
        }