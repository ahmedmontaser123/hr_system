from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request
from schemas.api_schemas import (
    CreateSessionRequest,
    SessionResponse,
    QuestionResponse,
    AnswerResponse,
    EvaluationResult,
    SummaryResponse,
)
from interview import InterviewSession
from llm import QuestionsGenerator, Evaluator, ClassificationQuestion, Transcript

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_store(request: Request) -> dict:
    return request.app.state.sessions


@router.post("", response_model=SessionResponse, status_code=201)
def create_session(body: CreateSessionRequest, request: Request):
    store = _get_store(request)
    session_id = str(uuid4())
    session = InterviewSession(
        transcript=request.app.state.transcript,
        generator=QuestionsGenerator(request.app.state.chains),
        classifier=ClassificationQuestion(request.app.state.chains),
        evaluator=Evaluator(request.app.state.chains),
    )
    store[session_id] = {
        "session": session,
        "role": body.role,
        "skills": body.skills,
        "results": [],
    }
    return SessionResponse(
        id=session_id,
        role=body.role,
        skills=body.skills,
    )


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    return SessionResponse(
        id=session_id,
        role=entry["role"],
        skills=entry["skills"],
        current_question=sess.current_question,
        current_category=sess.current_category,
        question_history=sess.question_history,
        results_count=len(entry["results"]),
    )


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, request: Request):
    store = _get_store(request)
    if session_id not in store:
        raise HTTPException(404, "Session not found")
    del store[session_id]


@router.post("/{session_id}/questions", response_model=QuestionResponse)
def generate_question(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    sess.generate_question(entry["role"], entry["skills"])
    sess.classify_current_question()
    return QuestionResponse(
        question=sess.current_question,
        category=sess.current_category,
    )


@router.get("/{session_id}/current-question", response_model=QuestionResponse)
def get_current_question(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    if not sess.current_question:
        raise HTTPException(404, "No question generated yet")
    return QuestionResponse(
        question=sess.current_question,
        category=sess.current_category,
    )


@router.post("/{session_id}/answers", response_model=AnswerResponse)
async def submit_answer(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    audio_bytes = await request.body()
    result = sess.evaluate_answer(audio_bytes)
    entry["results"].append(result)
    ev = result.get("evaluation", {})
    return AnswerResponse(
        question=result["question"],
        category=result.get("category"),
        transcript=result.get("transcript", ""),
        evaluation=EvaluationResult(
            score=ev.get("score"),
            feedback=ev.get("feedback"),
            status=ev.get("status"),
            message=ev.get("message"),
        ),
    )


@router.get("/{session_id}/answers", response_model=list[AnswerResponse])
def get_answers(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    results = []
    for r in entry["results"]:
        ev = r.get("evaluation", {})
        results.append(AnswerResponse(
            question=r["question"],
            category=r.get("category"),
            transcript=r.get("transcript", ""),
            evaluation=EvaluationResult(
                score=ev.get("score"),
                feedback=ev.get("feedback"),
                status=ev.get("status"),
                message=ev.get("message"),
            ),
        ))
    return results


@router.get("/{session_id}/summary", response_model=SummaryResponse)
def get_summary(session_id: str, request: Request):
    store = _get_store(request)
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(404, "Session not found")
    sess: InterviewSession = entry["session"]
    summary = sess.finish(entry["results"])
    results = []
    for r in summary.get("results", entry["results"]):
        ev = r.get("evaluation", {})
        results.append(AnswerResponse(
            question=r["question"],
            category=r.get("category"),
            transcript=r.get("transcript", ""),
            evaluation=EvaluationResult(
                score=ev.get("score"),
                feedback=ev.get("feedback"),
                status=ev.get("status"),
                message=ev.get("message"),
            ),
        ))
    return SummaryResponse(
        total_questions=summary["total_questions"],
        evaluated=summary["evaluated"],
        final_score=summary["final_score"],
        results=results,
    )
