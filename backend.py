import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-vectors")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required. Add it to your local .env file.")
    return value


@lru_cache(maxsize=1)
def _chat_model() -> ChatOpenAI:
    _require_env("OPENAI_API_KEY")
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2)


@lru_cache(maxsize=1)
def _embedding_model() -> OpenAIEmbeddings:
    _require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _pinecone_index():
    api_key = _require_env("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def extract_text_from_pdf(pdf_path: str):
    """Extract text from a PDF file and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)


def extract_uploaded_resume(pdf_path: str) -> str:
    """Extract a plain-text resume from an uploaded PDF."""
    resume_documents = extract_text_from_pdf(pdf_path)
    return "\n".join(doc.page_content for doc in resume_documents)


def load_embeddings_from_json(json_file: str) -> int:
    """Load resume example embeddings from JSON and upsert them into Pinecone."""
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {json_file}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    vectors = [
        (file_name, entry["embedding"], {"content": entry["content"]})
        for file_name, entry in data.items()
    ]

    if vectors:
        _pinecone_index().upsert(vectors=vectors)

    return len(vectors)


def similarity_search(query_text: str, top_k: int = 3):
    """Perform a similarity search in Pinecone."""
    query_embedding = _embedding_model().embed_query(query_text)
    return _pinecone_index().query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )["matches"]


def fetch_chunk_content(vector_id: str) -> str:
    """Fetch the content of a specific vector using its ID."""
    result = _pinecone_index().fetch([vector_id])
    if result and vector_id in result.get("vectors", {}):
        return result["vectors"][vector_id].get("metadata", {}).get("content", "No content found.")
    return "No content found."


def _invoke(messages: list[BaseMessage]) -> str:
    response = _chat_model().invoke(messages)
    return response.content


def _format_chat_history(chat_history: list[dict[str, Any]]) -> list[BaseMessage]:
    formatted_messages: list[BaseMessage] = []
    for entry in chat_history:
        content = entry.get("message", "").strip()
        if not content:
            continue
        if entry.get("role") == "user":
            formatted_messages.append(HumanMessage(content=content))
        else:
            formatted_messages.append(AIMessage(content=content))
    return formatted_messages


def initial_status(resume_text: str, job_description: str) -> str:
    """Generate interview questions based on a resume and target job description."""
    prompt_content = f"""
You are an expert resume editor and interviewer. Analyze the following resume content:
{resume_text}

And the following job description:
{job_description}

Generate four interview questions targeting gaps in experience or skills. Each question
should be no more than 100 words.

Then generate two behavioral questions related to general professional experiences.

Return only the numbered questions, formatted as:
1.
2.
3.
4.
5.
6.
"""

    return _invoke([HumanMessage(content=prompt_content)])


conversation_history: list[BaseMessage] = []


def simplify_content(resume_content: str, job_description: str, user_query: str) -> str:
    """Create interview questions while preserving conversation context."""
    conversation_history.append(HumanMessage(content=user_query))

    system_prompt = SystemMessage(
        content=(
            "You are an expert resume editor. Analyze the resume content and job "
            "description to produce exactly four interview questions. Each question "
            "must be no more than 100 words."
        )
    )
    prompt_with_history = [
        system_prompt,
        HumanMessage(content=f"Resume:\n{resume_content}\n\nJob description:\n{job_description}"),
        *conversation_history,
    ]

    answer = _invoke(prompt_with_history)
    conversation_history.append(AIMessage(content=answer))
    return answer


def request_resume(
    resume_text: str,
    job_description: str,
    chat_history: list[dict[str, Any]],
    embedding: bool = True,
    similarity_search_result: list | None = None,
) -> str:
    """Generate a revised resume using the user's resume, target job, and interview record."""
    formatted_chat_history = _format_chat_history(chat_history)
    similarity_search_result = similarity_search_result or []

    prompt: list[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a professional resume editor. Revise the resume to fit the "
                "target job description using only information supported by the original "
                "resume and interview record."
            )
        ),
        HumanMessage(content="Original resume:\n" + resume_text),
        HumanMessage(content="Target job description:\n" + job_description),
    ]

    if embedding and similarity_search_result:
        examples = []
        for index, match in enumerate(similarity_search_result[:2], start=1):
            content = match.get("metadata", {}).get("content")
            if content:
                examples.append(f"Example {index}:\n{content}")
        if examples:
            prompt.append(
                HumanMessage(
                    content=(
                        "Use these examples only as writing style and structure references. "
                        "Do not copy candidate facts from them.\n\n" + "\n\n".join(examples)
                    )
                )
            )

    prompt.extend(
        [
            HumanMessage(
                content=(
                    "Interview history that can inform useful experience. Assume the "
                    "candidate's answers are true."
                )
            ),
            *formatted_chat_history,
            HumanMessage(
                content=(
                    "Provide only the revised resume content. Do not include extra "
                    "explanations or commentary."
                )
            ),
        ]
    )

    return _invoke(prompt)


def request_coverletter(resume_text: str, job_description: str, chat_history: list[dict[str, Any]]) -> str:
    """Generate a cover letter from the resume, target job, and interview record."""
    prompt = [
        SystemMessage(
            content=(
                "You are a professional cover letter editor. Write a tailored cover "
                "letter using the resume, job description, and mock interview record."
            )
        ),
        HumanMessage(content="Original resume:\n" + resume_text),
        HumanMessage(content="Target job description:\n" + job_description),
        HumanMessage(content="Interview history that can inform the cover letter:"),
        *_format_chat_history(chat_history),
        HumanMessage(content="Provide only the cover letter. Do not include extra explanations."),
    ]

    return _invoke(prompt)


def request_suggestion(resume_text: str, job_description: str, chat_history: list[dict[str, Any]]) -> str:
    """Suggest improvements for the user's interview performance and resume."""
    prompt = [
        SystemMessage(
            content=(
                "You are a professional interviewer and resume coach. Provide practical "
                "suggestions to improve interview performance and resume alignment."
            )
        ),
        HumanMessage(content="Original resume:\n" + resume_text),
        HumanMessage(content="Target job description:\n" + job_description),
        HumanMessage(content="Interview history:"),
        *_format_chat_history(chat_history),
        HumanMessage(
            content=(
                "Provide focused suggestions based on the interview process, original "
                "resume, and job description."
            )
        ),
    ]

    return _invoke(prompt)


def addition_question(
    resume_text: str,
    job_description: str,
    chat_history: list[dict[str, Any]],
    query: str,
) -> str:
    """Answer a user's follow-up question after the mock interview."""
    prompt = [
        SystemMessage(
            content=(
                "You are a professional interviewer. Use the resume, job description, "
                "and mock interview record to answer the candidate's follow-up question."
            )
        ),
        HumanMessage(content="Original resume:\n" + resume_text),
        HumanMessage(content="Target job description:\n" + job_description),
        HumanMessage(content="Interview history:"),
        *_format_chat_history(chat_history),
        HumanMessage(content=query),
    ]

    return _invoke(prompt)
