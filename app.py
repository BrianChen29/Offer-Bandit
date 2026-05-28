import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import backend
from htmlTemplates import bot_template, css, user_template


load_dotenv()

EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "embeddings_with_metadata.json")
USE_RESUME_EXAMPLES = os.getenv("USE_RESUME_EXAMPLES", "false").lower() in {"1", "true", "yes"}


def initialize_state() -> None:
    defaults = {
        "chat_started": False,
        "chat_history": [],
        "conversion_count": 0,
        "user_input": "",
        "interview_just_end": True,
        "interview_records": [],
        "resume_text": "",
        "job_description": "",
        "questions": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def process_resumes(pdf_docs) -> str:
    """Extract text from uploaded PDFs without leaving local copies behind."""
    if not pdf_docs:
        raise ValueError("Please upload at least one resume PDF.")

    extracted_resumes = []
    for pdf in pdf_docs:
        suffix = Path(pdf.name).suffix or ".pdf"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(pdf.getbuffer())
                tmp_path = tmp_file.name

            extracted_resumes.append(backend.extract_uploaded_resume(tmp_path))
            st.success(f"Processed {pdf.name}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return "\n\n".join(extracted_resumes)


def parse_questions(initial_content: str) -> list[str]:
    """Parse numbered interview questions from the model response."""
    questions = re.findall(r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)", initial_content, re.DOTALL)
    return [question.strip() for question in questions if question.strip()]


def generate_initial_questions(resume_text: str, job_description: str) -> list[str]:
    initial_content = backend.initial_status(resume_text, job_description)
    questions = parse_questions(initial_content)

    if not questions:
        raise RuntimeError("The model did not return numbered interview questions. Please try again.")

    return questions


def display_chat_history() -> None:
    for chat in st.session_state.chat_history:
        template = user_template if chat["role"] == "user" else bot_template
        st.write(template.replace("{{MSG}}", chat["message"]), unsafe_allow_html=True)


def ask_next_question() -> None:
    if st.session_state.conversion_count < len(st.session_state.questions):
        st.session_state.chat_history.append(
            {
                "role": "bot",
                "message": st.session_state.questions[st.session_state.conversion_count],
            }
        )
        st.session_state.conversion_count += 1


def handle_user_input() -> None:
    user_query = st.session_state.user_input.strip()
    if not user_query:
        return

    st.session_state.chat_history.append({"role": "user", "message": user_query})
    ask_next_question()
    st.session_state.user_input = ""


def after_questions() -> None:
    user_query = st.session_state.user_input.strip()

    if st.session_state.interview_just_end:
        if user_query:
            st.session_state.chat_history.append({"role": "user", "message": user_query})
        st.session_state.interview_just_end = False
        st.session_state.interview_records = list(st.session_state.chat_history)
        st.session_state.chat_history.append(
            {
                "role": "bot",
                "message": (
                    "Thank you for participating. All technical and behavioral questions are complete. "
                    "You can ask follow-up questions, or press Finish Interview to generate your revised "
                    "resume, cover letter, and suggestions."
                ),
            }
        )
    elif user_query:
        response = backend.addition_question(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.chat_history,
            user_query,
        )
        st.session_state.chat_history.append({"role": "user", "message": user_query})
        st.session_state.chat_history.append({"role": "bot", "message": response})

    st.session_state.user_input = ""


def finish_process() -> None:
    similarity_search_result = []

    if USE_RESUME_EXAMPLES:
        try:
            backend.load_embeddings_from_json(EMBEDDINGS_FILE)
            similarity_search_result = backend.similarity_search(st.session_state.resume_text, top_k=2)
        except Exception as exc:
            st.warning(f"Resume example search is unavailable, continuing without examples: {exc}")

    with st.spinner("Generating application materials..."):
        revised_resume = backend.request_resume(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.interview_records,
            USE_RESUME_EXAMPLES and bool(similarity_search_result),
            similarity_search_result,
        )
        generated_coverletter = backend.request_coverletter(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.interview_records,
        )
        generated_suggestion = backend.request_suggestion(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.interview_records,
        )

    st.session_state.chat_history.extend(
        [
            {"role": "bot", "message": "Revised resume:\n" + revised_resume},
            {"role": "bot", "message": "Cover letter:\n" + generated_coverletter},
            {"role": "bot", "message": "Suggestions:\n" + generated_suggestion},
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Offer Bandit", page_icon=":briefcase:")
    st.write(css, unsafe_allow_html=True)
    initialize_state()

    with st.sidebar:
        st.subheader("Resume Setup")
        pdf_docs = st.file_uploader("Upload resume PDF", accept_multiple_files=True, type=["pdf"])
        st.session_state.job_description = st.text_area("Target job description")

        if st.button("Prepare Interview"):
            with st.spinner("Preparing your mock interview..."):
                try:
                    st.session_state.resume_text = process_resumes(pdf_docs)
                    st.session_state.questions = generate_initial_questions(
                        st.session_state.resume_text,
                        st.session_state.job_description,
                    )
                    st.session_state.chat_history = []
                    st.session_state.conversion_count = 0
                    st.session_state.chat_started = False
                    st.session_state.interview_just_end = True
                    st.success("Interview questions are ready.")
                except Exception as exc:
                    st.error(str(exc))

    st.header("Offer Bandit Interview Assistant")

    if st.button("Start Interview") and st.session_state.questions and st.session_state.conversion_count == 0:
        st.session_state.chat_started = True
        st.session_state.chat_history.append(
            {
                "role": "bot",
                "message": (
                    "You will be asked four experience-based questions and two behavioral questions. "
                    "Please answer with concrete details such as company, timeline, skills, and impact."
                ),
            }
        )
        ask_next_question()

    if st.session_state.chat_started:
        display_chat_history()
        st.text_input("Your answer or follow-up question", key="user_input")

        if st.session_state.conversion_count < len(st.session_state.questions):
            st.button("Submit", on_click=handle_user_input)
        else:
            st.button("Submit", on_click=after_questions)
            if not st.session_state.interview_just_end:
                st.button("Finish Interview", on_click=finish_process)


if __name__ == "__main__":
    main()
