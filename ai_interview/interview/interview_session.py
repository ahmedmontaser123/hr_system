from audio.speech_to_text import WhisperLoader
from llm import QuestionsGenerator, Evaluator,ClassficationQuestion


class InterviewSession:
    def __init__(
        self,
        whisper: WhisperLoader,
        generator: QuestionsGenerator,
        classfier: ClassficationQuestion,
        evaluator: Evaluator
    ):
        self.whisper = whisper
        self.generator = generator
        self.classfier = classfier
        self.evaluator = evaluator

        self.current_question = None
        self.current_category = None

    # =========================
    # 1. Generate Question
    # =========================
    def generate_question(self, role: str, skills: str) -> dict:
        result = self.generator.generate(role=role, description=skills)
        self.current_question = result
        return result
    
    def classfied_question(self):
        result = self.classfier.classify(self.current_question)
        self.current_category = result

        return result


        

    def evaluate_answer(self, audio_bytes: bytes, suffix: str = ".wav") -> dict:

        if not self.current_question:
            return {
                "status": "error",
                "message": "No active question"
            }

        try:
            transcript = self.whisper.transcribe(audio_bytes, suffix)
        except Exception as e:
            return {
                "question": self.current_question,
                "transcript": "",
                "evaluation": {
                    "status": "error",
                    "message": f"Audio transcription failed: {str(e)}"
                }
            }

        if not transcript or len(transcript.split()) < 3:
            return {
                "question": self.current_question,
                "transcript": transcript,
                "evaluation": {
                    "status": "invalid_answer",
                    "message": "Answer too short"
                }
            }

        evaluation = self.evaluator.evaluate_answer(
            question=self.current_question,
            category=self.current_category,
            answer=transcript
            )
        return {
            "question": self.current_question,
            "category": self.current_category,
            "transcript": transcript,
            "evaluation": evaluation
        }
    


    def finish(self, results: list) -> dict:
        evaluated = [
         r for r in results
         if "score" in r.get("evaluation", {})
         ]

        if not evaluated:
            return {
            "total_questions": len(results),
            "evaluated": 0,
            "final_score": 0,
            "summary": "No valid answers were evaluated"
             }

        scores = [
        r["evaluation"]["score"]
        for r in evaluated
        ]

        final_score = round(sum(scores) / len(scores), 2)

        return {
        "total_questions": len(results),
        "evaluated": len(evaluated),
        "final_score": f"{final_score}/10",  # ← أوضح للمستخدم
        "results": results
    }
    


