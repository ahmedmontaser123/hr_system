from pydantic import BaseModel
from typing import Optional


class CreateSessionRequest(BaseModel):
    role: str
    skills: str


class SessionResponse(BaseModel):
    id: str
    role: str
    skills: str
    current_question: Optional[str] = None
    current_category: Optional[str] = None
    question_history: list[str] = []
    results_count: int = 0


class QuestionResponse(BaseModel):
    question: str
    category: str


class EvaluationResult(BaseModel):
    score: Optional[int] = None
    feedback: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None


class AnswerResponse(BaseModel):
    question: str
    category: Optional[str] = None
    transcript: str
    evaluation: EvaluationResult


class SummaryResponse(BaseModel):
    total_questions: int
    evaluated: int
    final_score: int
    results: list[AnswerResponse] = []
