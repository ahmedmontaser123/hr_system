# interview/interview_session.py
from audio.speech_to_text import WhisperLoader
from llm import QuestionsGenerator, Evaluator

class InterviewSession:
    def __init__(self, whisper: WhisperLoader, generator: QuestionsGenerator, evaluator: Evaluator):
        self.whisper = whisper
        self.generator = generator
        self.evaluator = evaluator

    def answer(self, question: str, audio_bytes: bytes) -> dict:
        transcript = self.whisper.transcribe(audio_bytes)
        evaluation = self.evaluator.process(question, transcript)
        return {
            "question": question,
            "transcript": transcript,
            "evaluation": evaluation
        }

    def finish(self, results: list) -> dict:
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