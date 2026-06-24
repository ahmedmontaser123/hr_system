import streamlit as st
from helpers import get_settings
from llm import QuestionsGenerator, Evaluator, ClassficationQuestion
from llm.providers.ollama_provider import OllamaProvider
from llm.chains import Chains
from audio.speech_to_text import WhisperLoader
from interview import InterviewSession
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="AI Interview", layout="centered")

settings = get_settings()


@st.cache_resource
def load_llm():
    provider = OllamaProvider(settings)
    return provider.get_llm()


@st.cache_resource
def load_whisper():
    return WhisperLoader(settings)


llm = load_llm()
whisper = load_whisper()
chains = Chains(llm)

if "session" not in st.session_state:
    st.session_state.session = InterviewSession(
        whisper=whisper,
        generator=QuestionsGenerator(chains),
        classfier=ClassficationQuestion(chains),
        evaluator=Evaluator(chains),
    )
    st.session_state.results = []
    st.session_state.phase = "setup"
    st.session_state.recording_id = 0

st.title("AI Interview")

with st.sidebar:
    st.header("Configuration")
    role = st.text_input("Role", value="Software Engineer")
    skills = st.text_area("Required Skills", value="Python, FastAPI, PostgreSQL")

    if st.button("Generate Question", use_container_width=True):
        with st.spinner("Generating question..."):
            st.session_state.session.generate_question(role, skills)
            st.session_state.session.classfied_question()
            st.session_state.phase = "answering"
            st.session_state.recording_id += 1
            st.rerun()

    if st.button("Show Summary", use_container_width=True, type="primary"):
        st.session_state.phase = "summary"
        st.rerun()

question = st.session_state.session.current_question
category = st.session_state.session.current_category

if st.session_state.phase == "setup":
    st.info("Configure the role and skills, then click **Generate Question**.")

elif st.session_state.phase == "answering" and question:
    st.subheader("Question")
    st.markdown(f"**Category:** `{category}`")
    st.info(question)

    audio_bytes = audio_recorder(
        key=f"recorder_{st.session_state.recording_id}",
        text="Record your answer",
        recording_color="#e74c3c",
        neutral_color="#6c757d",
    )

    if audio_bytes:
        with st.spinner("Transcribing and evaluating..."):
            result = st.session_state.session.evaluate_answer(audio_bytes)

        status = result["evaluation"]["status"]

        if status == "evaluated":
            st.success("Answer evaluated")
            col1, col2 = st.columns(2)
            col1.metric("Score", f"{result['evaluation']['evaluation']['score']}/10")
            col2.metric("Transcript length", f"{len(result['transcript'].split())} words")
            st.markdown(f"**Transcript:** {result['transcript']}")
            st.markdown(f"**Feedback:** {result['evaluation']['evaluation']['feedback']}")
            st.session_state.results.append(result)
            st.session_state.phase = "done"
            st.rerun()
        elif status == "invalid_answer":
            st.warning("Answer too short. Please provide a more detailed answer.")
        else:
            st.error(f"Error: {result['evaluation'].get('message', 'Unknown error')}")

elif st.session_state.phase == "done":
    st.success("Answer recorded!")
    if st.button("Next Question", use_container_width=True):
        st.session_state.session.current_question = None
        st.session_state.session.current_category = None
        st.session_state.phase = "setup"
        st.rerun()

if st.session_state.phase == "summary":
    st.subheader("Interview Summary")
    results = st.session_state.results
    if results:
        summary = st.session_state.session.finish(results)
        st.metric("Total Questions", summary["total_questions"])
        st.metric("Evaluated", summary["evaluated"])
        st.metric("Final Score", summary["final_score"])

        for i, r in enumerate(results, 1):
            with st.expander(f"Q{i}: {r['question'][:80]}..."):
                st.markdown(f"**Question:** {r['question']}")
                st.markdown(f"**Category:** {r.get('category', 'N/A')}")
                st.markdown(f"**Transcript:** {r.get('transcript', 'N/A')}")
                if r["evaluation"]["status"] == "evaluated":
                    ev = r["evaluation"]["evaluation"]
                    st.markdown(f"**Score:** {ev['score']}/10")
                    st.markdown(f"**Feedback:** {ev['feedback']}")

        if st.button("New Interview", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()
    else:
        st.warning("No questions were answered.")
