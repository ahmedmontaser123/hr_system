import streamlit as st
from helpers import Settings
from ai import LLMLoader, build_chains, Evaluator, QuestionGenerator
from audio.speech_to_text import WhisperLoader
from interview.interview_session import InterviewSession

st.title("🎤 AI Interview Assistant")


@st.cache_resource(show_spinner="Loading models... (first time only)")
def load_models():
    """يتحمّل مرة واحدة فعليًا على مستوى التطبيق كله، مش لكل session."""
    settings = Settings()
    llm_loader = LLMLoader(settings)
    whisper_loader = WhisperLoader(settings)
    chains = build_chains(llm_loader)

    evaluator = Evaluator(chains)
    question_generator = QuestionGenerator(chains)

    return evaluator, question_generator, whisper_loader


evaluator, question_generator, whisper_loader = load_models()
st.success("Models ready ✅")

# بدء session جديدة
if "interview_session" not in st.session_state:
    role = st.text_input("Role", "Backend Developer")
    description = st.text_area("Job Description", "Python, REST APIs, databases")

    if st.button("Start Interview"):
        st.session_state.interview_session = InterviewSession(
            candidate_name="Candidate",
            role=role,
            description=description,
        )
        st.rerun()

else:
    session = st.session_state.interview_session

    if "current_question" not in st.session_state:
        if st.button("Generate Question"):
            question = question_generator.generate(
                role=session.role,
                description=session.description,
                topic="general",
                difficulty=session.current_difficulty,
                previous_topics=session.get_covered_topics(),
            )
            st.session_state.current_question = question
            st.rerun()

    else:
        st.write("**Question:**", st.session_state.current_question)

        audio_file = st.file_uploader("Upload your answer (audio)", type=["wav", "mp3", "ogg"])

        if audio_file and st.button("Submit Answer"):
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.read())

            answer_text = whisper_loader.transcribe("temp_audio.wav")
            st.write("**Transcribed Answer:**", answer_text)

            result = evaluator.process(
                question=st.session_state.current_question,
                answer=answer_text,
            )
            st.json(result)

            if result["status"] == "evaluated":
                session.add_record(
                    question=st.session_state.current_question,
                    answer=answer_text,
                    topic="general",
                    difficulty=session.current_difficulty,
                    evaluation=result["evaluation"],
                    relevance=result["relevance"],
                    question_type=result["question_type"],
                )

            del st.session_state.current_question
            st.rerun()

    if st.button("Show Summary"):
        st.json(session.get_summary())
