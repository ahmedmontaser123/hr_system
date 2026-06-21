# interview/interview_session.py
from audio.speech_to_text import WhisperLoader
from llm import QuestionsGenerator
from llm import Evaluator

class InterviewSession:
    def __init__(self, whisper: WhisperLoader, generator: QuestionsGenerator, evaluator: Evaluator):
        self.whisper = whisper
        self.generator = generator
        self.evaluator = evaluator

    def start(self, role: str, description: str, num_questions: int = 5):
        """يتكال مرة واحدة في الأول — بيولد الأسئلة"""
        questions = []
        for _ in range(num_questions):
            question = self.generator.generate(role, description)
            questions.append(question)
        return questions

    def answer(self, question: str, audio_bytes: bytes) -> dict:
        """بياخد audio ويرجع التقييم"""
        transcript = self.whisper.transcribe(audio_bytes)
        evaluation = self.evaluator.process(question, transcript)
        return {
            "question": question,
            "transcript": transcript,
            "evaluation": evaluation
        }

    def finish(self, results: list) -> dict:
        """summary"""
        evaluated = [r for r in results if r["evaluation"]["status"] == "evaluated"]
        
        if not evaluated:
            return {"final_score": 0, "summary": "No valid answers"}

        scores = []
        for r in evaluated:
            ev = r["evaluation"]["evaluation"]
            score = sum(v for v in ev.values() if isinstance(v, (int, float)))
            scores.append(score)

        avg = sum(scores) / len(scores)

        return {
            "total_questions": len(results),
            "evaluated": len(evaluated),
            "final_score": round(avg, 2),
            "results": results
        }