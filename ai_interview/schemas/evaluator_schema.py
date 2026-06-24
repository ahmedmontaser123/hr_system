from pydantic import BaseModel ,Field

class EvaluationSchema(BaseModel):
    score: int = Field(
        ge=0,
        le=10,
        description="Overall score from 0 to 10"
    )
    feedback: str = Field(
        description="Short structured feedback"
    )