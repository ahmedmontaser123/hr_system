# main.py
import streamlit as st
from helpers import get_settings
from llm import build_chains
from llm import QuestionsGenerator
from llm import Evaluator
from llm.providers.hugging_face_providers import HuggingFaceProvider
from audio.speech_to_text import WhisperLoader
from interview.interview_session import InterviewSession

st.title("🎤 AI Interview Assistant")

@st.cache_resource(show_spinner="Loading models... (first time only)")
def load_models():
    settings = get_settings()
    llm_provider = HuggingFaceProvider(settings)
    whisper = WhisperLoader(settings)
    chains = build_chains(llm_provider.get_llm())
    generator = QuestionsGenerator(chains)
    evaluator = Evaluator(chains)
    
    
    return whisper, generator, evaluator

whisper, generator, evaluator = load_models()
st.success("Models ready ✅")

if "session" not in st.session_state:
    role = st.text_input("Role", "Backend Developer")
    description = st.text_area("Job Description", "Python, REST APIs, databases")

    if st.button("Start Interview"):
        session = InterviewSession(whisper, generator, evaluator)
        questions = session.start(role, description, num_questions=5)
        st.session_state.session = session
        st.session_state.questions = questions
        st.session_state.current_index = 0
        st.session_state.results = []
        st.rerun()

else:
    session = st.session_state.session
    questions = st.session_state.questions
    index = st.session_state.current_index

    if index < len(questions):
        st.progress(index / len(questions))
        st.write(f"**Question {index + 1}/{len(questions)}:**")
        st.write(questions[index])

        audio = st.audio_input("Record your answer")

        if audio and st.button("Submit Answer"):
            with st.spinner("Processing..."):
                result = session.answer(questions[index], audio.read())

            st.write("**Your answer:**", result["transcript"])
            st.json(result["evaluation"])

            st.session_state.results.append(result)
            st.session_state.current_index += 1
            st.rerun()

    else:
        st.success("Interview finished! 🎉")
        summary = session.finish(st.session_state.results)
        st.json(summary)

        if st.button("Start New Interview"):
            for key in ["session", "questions", "current_index", "results"]:
                del st.session_state[key]
            st.rerun()