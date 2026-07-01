import struct
import sys
from fastapi import FastAPI
from fastapi.testclient import TestClient

from helpers import get_settings
from llm.chains import Chains
from llm import QuestionsGenerator, Evaluator, ClassificationQuestion, Transcript
from routers.sessions import router


def _generate_silence_wav(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * 2
    file_size = 36 + data_size
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', file_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b'data', data_size,
    )
    samples = struct.pack(f'<{num_samples}h', *([0] * num_samples))
    return header + samples


def create_app() -> FastAPI:
    settings = get_settings()

    from llm.providers.ollama_provider import OllamaProvider
    provider = OllamaProvider(settings)

    from llm.providers.faster_whisper_provider import WhisperLoader
    chains = Chains(provider.get_llm())
    whisper = WhisperLoader(settings)
    transcript = Transcript(whisper)

    app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
    app.state.sessions = {}
    app.state.chains = chains
    app.state.transcript = transcript
    app.include_router(router)
    return app


def test_endpoints():
    print("\n=== Creating FastAPI app (loading models, this may take a while) ===\n")
    try:
        app = create_app()
    except Exception as e:
        print(f"Failed to create app: {e}")
        sys.exit(1)

    client = TestClient(app)
    base = "/sessions"
    passed = 0
    failed = 0

    def check(name: str, status: int, resp, expect_body: bool = True):
        nonlocal passed, failed
        ok = resp.status_code == status
        label = "✅" if ok else "❌"
        extra = f" (got {resp.status_code})" if not ok else ""
        print(f"  {label} {name}{extra}")
        if ok:
            passed += 1
        else:
            failed += 1

    # ── 1. POST /sessions → 201 ──
    resp = client.post(base, json={"role": "Software Engineer", "skills": "Python, FastAPI, SQL"})
    check("POST /sessions  ->  201", 201, resp)
    if resp.status_code != 201:
        print("\nCannot continue – session creation failed")
        sys.exit(1)
    sid = resp.json()["id"]

    # ── 2. GET /sessions/{id} → 200 ──
    resp = client.get(f"{base}/{sid}")
    check("GET  /sessions/{{id}}   ->  200", 200, resp)

    # ── 3. GET /sessions/{id} → 404 ──
    resp = client.get(f"{base}/does-not-exist")
    check("GET  /sessions/{{id}}   ->  404 (no session)", 404, resp)

    # ── 4. DELETE /sessions/{id} → 404 ──
    resp = client.delete(f"{base}/does-not-exist")
    check("DEL  /sessions/{{id}}   ->  404 (no session)", 404, resp)

    # ── 5. POST /sessions/{id}/questions → 200 ──
    print("  ⏳  Generating question (LLM call – may be slow) ...")
    resp = client.post(f"{base}/{sid}/questions")
    check("POST /sessions/{{id}}/questions  ->  200", 200, resp)

    # ── 6. GET /sessions/{id}/current-question → 200 ──
    resp = client.get(f"{base}/{sid}/current-question")
    check("GET  /sessions/{{id}}/current-question  ->  200", 200, resp)

    # ── 7. GET /sessions/{id}/current-question → 404 (no session) ──
    resp = client.get(f"{base}/does-not-exist/current-question")
    check("GET  /sessions/{{id}}/current-question  ->  404 (no session)", 404, resp)

    # ── 8. GET /sessions/{id}/current-question → 404 (no question yet) ──
    # Create a second session but don't generate a question for it
    resp2 = client.post(base, json={"role": "Data Scientist", "skills": "ML, Python"})
    sid2 = resp2.json()["id"]
    resp = client.get(f"{base}/{sid2}/current-question")
    check("GET  /sessions/{{id}}/current-question  ->  404 (no question)", 404, resp)

    # ── 9. POST /sessions/{id}/answers → 200 ──
    wav = _generate_silence_wav()
    resp = client.post(f"{base}/{sid}/answers", content=wav)
    check("POST /sessions/{{id}}/answers  ->  200", 200, resp)

    # ── 10. POST /sessions/{id}/answers → 404 ──
    resp = client.post(f"{base}/does-not-exist/answers", content=wav)
    check("POST /sessions/{{id}}/answers  ->  404 (no session)", 404, resp)

    # ── 11. GET /sessions/{id}/answers → 200 ──
    resp = client.get(f"{base}/{sid}/answers")
    check("GET  /sessions/{{id}}/answers  ->  200", 200, resp)

    # ── 12. GET /sessions/{id}/answers → 404 ──
    resp = client.get(f"{base}/does-not-exist/answers")
    check("GET  /sessions/{{id}}/answers  ->  404 (no session)", 404, resp)

    # ── 13. GET /sessions/{id}/summary → 200 ──
    resp = client.get(f"{base}/{sid}/summary")
    check("GET  /sessions/{{id}}/summary  ->  200", 200, resp)

    # ── 14. GET /sessions/{id}/summary → 404 ──
    resp = client.get(f"{base}/does-not-exist/summary")
    check("GET  /sessions/{{id}}/summary  ->  404 (no session)", 404, resp)

    # ── 15. DELETE /sessions/{id} → 204 ──
    resp = client.delete(f"{base}/{sid}")
    check("DEL  /sessions/{{id}}         ->  204", 204, resp)

    # ── 16. DELETE /sessions/{id} → 404 (already deleted) ──
    resp = client.delete(f"{base}/{sid}")
    check("DEL  /sessions/{{id}}         ->  404 (already deleted)", 404, resp)

    total = passed + failed
    print(f"\n{'=' * 42}")
    print(f"  {passed} passed, {failed} failed  |  {total} total")
    print(f"{'=' * 42}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    test_endpoints()
