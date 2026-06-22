import streamlit as st
import logging
from helpers import get_settings
from llm import build_chains, QuestionsGenerator, Evaluator
from llm.providers.ollama_provider import OllamaProvider
from audio.speech_to_text import WhisperLoader
from interview.interview_session import InterviewSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

st.title("🎤 AI Interview Test")

@st.cache_resource(show_spinner="Loading models...")
def load():
    logger.info("Loading Whisper...")
    settings = get_settings()
    whisper = WhisperLoader(settings)
    logger.info("Whisper loaded ✅")

    logger.info("Loading LLM...")
    llm_provider = OllamaProvider(settings)
    chains = build_chains(llm_provider.get_llm())
    generator = QuestionsGenerator(chains)
    evaluator = Evaluator(chains)
    logger.info("LLM loaded ✅")

    return whisper, generator, evaluator

whisper, generator, evaluator = load()
st.success("Models ready ✅")

if "session" not in st.session_state:
    role = st.text_input("Role", "Backend Developer")
    description = st.text_area("Description", "Python, REST APIs, databases")

    if st.button("Start Interview"):
        logger.info("Starting interview...")
        st.session_state.session = InterviewSession(whisper, generator, evaluator)
        st.session_state.role = role
        st.session_state.description = description
        st.session_state.current_index = 0
        st.session_state.results = []
        st.session_state.current_question = None
        logger.info("Session created ✅")
        st.rerun()

else:
    session = st.session_state.session
    index = st.session_state.current_index
    max_questions = 5

    if index < max_questions:
        st.progress(index / max_questions)

        if st.session_state.current_question is None:
            logger.info(f"Generating question {index + 1}...")
            with st.spinner("Generating question..."):
                st.session_state.current_question = generator.generate(
                    st.session_state.role,
                    st.session_state.description
                )
            logger.info(f"Question {index + 1} generated ✅")

        st.write(f"**Question {index + 1}/{max_questions}:**")
        st.write(st.session_state.current_question)

        audio = st.audio_input("Record your answer")

        if audio and st.button("Submit Answer"):
            logger.info("Transcribing audio...")
            with st.spinner("Transcribing..."):
                transcript = whisper.transcribe(audio.read())
            logger.info(f"Transcript: {transcript}")

            logger.info("Evaluating answer...")
            with st.spinner("Evaluating..."):
                result = session.answer(st.session_state.current_question, audio.read())
            logger.info(f"Evaluation done ✅: {result['evaluation']['status']}")

            st.write("**Your answer:**", result["transcript"])
            st.json(result["evaluation"])

            st.session_state.results.append(result)
            st.session_state.current_index += 1
            st.session_state.current_question = None
            st.rerun()

    else:
        logger.info("Interview finished, generating summary...")
        st.success("Interview finished! 🎉")
        summary = session.finish(st.session_state.results)
        st.json(summary)
        logger.info(f"Final score: {summary['final_score']}")

        if st.button("Start New Interview"):
            for key in ["session", "role", "description", "current_index", "results", "current_question"]:
                del st.session_state[key]
            st.rerun()