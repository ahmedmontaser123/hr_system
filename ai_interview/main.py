import streamlit as st
import logging

from helpers import get_settings
from llm import build_chains, QuestionsGenerator, Evaluator
from llm.providers.ollama_provider import OllamaProvider
from audio.speech_to_text import WhisperLoader
from interview.interview_session import InterviewSession


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Interview Test",
    page_icon="🎤"
)

st.title("🎤 AI Interview Test")


@st.cache_resource(show_spinner="Loading models...")
def load():
    logger.info("Loading settings...")

    settings = get_settings()

    logger.info("Loading Whisper...")
    whisper = WhisperLoader(settings)

    # تحميل Whisper فعليًا عند بدء التطبيق
    whisper._get_model()

    logger.info("Whisper loaded ✅")

    logger.info("Loading Ollama...")
    llm_provider = OllamaProvider(settings)

    logger.info("Building chains...")
    chains = build_chains(llm_provider.get_llm())

    generator = QuestionsGenerator(chains)
    evaluator = Evaluator(chains)

    logger.info("LLM loaded ✅")

    return whisper, generator, evaluator


try:
    whisper, generator, evaluator = load()
    st.success("Models ready ✅")

except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()


# =========================
# START INTERVIEW
# =========================

if "session" not in st.session_state:

    role = st.text_input(
        "Role",
        "Backend Developer"
    )

    description = st.text_area(
        "Description",
        "Python, REST APIs, databases"
    )

    if st.button("Start Interview"):

        try:
            logger.info("Creating interview session...")

            st.session_state.session = InterviewSession(
                whisper,
                generator,
                evaluator
            )

            st.session_state.role = role
            st.session_state.description = description
            st.session_state.current_index = 0
            st.session_state.results = []
            st.session_state.current_question = None

            logger.info("Session created ✅")

            st.rerun()

        except Exception as e:
            logger.exception("Session creation failed")
            st.error(str(e))


# =========================
# INTERVIEW FLOW
# =========================

else:

    session = st.session_state.session
    index = st.session_state.current_index

    MAX_QUESTIONS = 5

    if index < MAX_QUESTIONS:

        st.progress(index / MAX_QUESTIONS)

        # Generate Question
        if st.session_state.current_question is None:

            try:
                logger.info(
                    f"Generating question {index + 1}"
                )

                with st.spinner(
                    "Generating question..."
                ):

                    question = generator.generate(
                        st.session_state.role,
                        st.session_state.description
                    )

                st.session_state.current_question = question

                logger.info(
                    f"Question {index + 1} generated ✅"
                )

            except Exception as e:
                logger.exception(
                    "Question generation failed"
                )
                st.error(str(e))
                st.stop()

        st.write(
            f"### Question {index + 1}/{MAX_QUESTIONS}"
        )

        st.write(
            st.session_state.current_question
        )

        audio = st.audio_input(
            "Record your answer"
        )

        if audio and st.button("Submit Answer"):

            try:

                logger.info(
                    "Submit clicked"
                )

                audio_bytes = audio.read()

                logger.info(
                    f"Audio received: {len(audio_bytes)} bytes"
                )

                with st.spinner(
                    "Processing answer..."
                ):

                    result = session.answer(
                        st.session_state.current_question,
                        audio_bytes
                    )

                logger.info(
                    "Answer processed successfully"
                )

                st.subheader("Transcript")

                st.write(
                    result.get(
                        "transcript",
                        "No transcript"
                    )
                )

                st.subheader("Evaluation")

                st.json(
                    result.get(
                        "evaluation",
                        {}
                    )
                )

                st.session_state.results.append(
                    result
                )

                st.session_state.current_index += 1

                st.session_state.current_question = None

                st.rerun()

            except Exception as e:

                logger.exception(
                    "Answer processing failed"
                )

                st.error(
                    f"Processing Error: {str(e)}"
                )

    else:

        try:

            logger.info(
                "Generating final summary..."
            )

            summary = session.finish(
                st.session_state.results
            )

            st.success(
                "Interview finished! 🎉"
            )

            st.json(summary)

            logger.info(
                f"Final score: {summary.get('final_score')}"
            )

        except Exception as e:

            logger.exception(
                "Summary generation failed"
            )

            st.error(str(e))

        if st.button(
            "Start New Interview"
        ):

            for key in [
                "session",
                "role",
                "description",
                "current_index",
                "results",
                "current_question"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()