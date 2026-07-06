# Offer Bandit

Offer Bandit is a RAG-enabled AI career assistant that helps job seekers turn a resume and target job description into a mock interview, a revised resume, a cover letter, and concrete improvement suggestions.

This repository is cleaned for portfolio review: API keys, private resumes, generated embeddings, and audio files are kept out of git. The core app uses OpenAI models for interview generation and writing assistance. Pinecone-backed resume examples are optional, and the voice interview work is kept as an optional STT/TTS prototype.

## Features

- Upload one or more resume PDFs.
- Paste a target job description.
- Generate four role-specific interview questions and two behavioral questions.
- Collect mock interview answers through a chat-style UI.
- Generate a revised resume, tailored cover letter, and improvement suggestions.
- Optionally use a Pinecone vector index of example resumes as writing-style references.
- Run optional speech utilities for Whisper-based STT and Coqui TTS-based response playback.

## AI/ML Engineering Highlights

- Retrieval-augmented generation pipeline over resume examples using OpenAI embeddings and Pinecone.
- Prompt orchestration that combines resume text, target job descriptions, interview answers, and retrieved examples.
- Privacy-aware project packaging with local-only `.env`, resume PDFs, embedding JSON, and generated audio artifacts.
- Optional voice prototype showing how the mock interview could evolve into a spoken interaction loop.

## Dataset & Evaluation (course project)

This repository is the cleaned code. The original CSCI 566 team project ("The Dropouts," 4 people) was backed by a curated dataset and a manual evaluation:

- **Dataset:** 250+ resumes (~200 well-revised used as embedding data, 50 incomplete used as testing data) and 400+ job descriptions spanning startups, unicorns, and enterprises. Resumes were sourced from a tech job-seeker community and de-identified (names/schools/companies/timelines removed) before embedding. The testing set was kept distinct from the embedding set.
- **Embedding ablation:** the team compared generated resumes with vs. without example-resume retrieval, scored on a 5-criteria 1–10 rubric (relevance, content accuracy, language/clarity, formatting/readability, specificity/impact). Embedding-enabled outputs averaged highest (GPT-4o + embeddings ≈ 9.4 vs. 7.98 for the original resume), with the largest gains in relevance (+~32%) and impact (+~22%).
- **Model selection by latency:** GPT-4 averaged >7.5s per question (>10s with TTS/STT), too slow for a real-time mock interview, so the system uses GPT-4o / GPT-4o-mini (<3s per question even with the voice modules).

*Note: the evaluation was an internal, manual assessment by the team on a small sample — useful as a directional signal, not a claim of real-user validation. The system is a functional prototype and was not deployed.*

## Project Structure

```text
.
├── app.py                  # Streamlit UI and interview flow
├── backend.py              # OpenAI, Pinecone, PDF extraction, and prompt orchestration
├── embedding_resume.py     # Utility script for generating local resume example embeddings
├── speech_tools.py         # Optional STT/TTS CLI utilities
├── db.py                   # Optional local FAISS helper functions
├── htmlTemplates.py        # Chat UI templates
├── requirements.txt
├── requirements-audio.txt
├── .env.example
└── .gitignore
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Add your local API keys to `.env`:

```text
OPENAI_API_KEY=...
PINECONE_API_KEY=... # optional unless USE_RESUME_EXAMPLES=true
```

The `.env` file is ignored by git and should never be committed.

## Run

```bash
streamlit run app.py
```

The default workflow does not require the resume embedding dataset. To enable example-resume retrieval, place a local `embeddings_with_metadata.json` in the project root and set:

```text
USE_RESUME_EXAMPLES=true
```

## Generate Resume Example Embeddings

Place `.docx` resumes in a local `Resume_Text_Preprocessing/` folder, then run:

```bash
python embedding_resume.py --input-dir Resume_Text_Preprocessing --output embeddings_with_metadata.json
```

Both `Resume_Text_Preprocessing/` and `embeddings_with_metadata.json` are ignored by git because they may contain private resume data.

## Optional STT/TTS Prototype

Install the optional audio dependencies only if you want to run the voice prototype:

```bash
pip install -r requirements-audio.txt
```

Generate speech from text:

```bash
python speech_tools.py tts --text "Tell me about a machine learning project you built." --output output.wav
```

Transcribe an audio file:

```bash
python speech_tools.py stt --audio output.wav
```

The Streamlit app currently uses text input for stability. The STT/TTS utilities preserve the voice-interview prototype without making the main demo depend on heavy audio packages.

## Portfolio Notes

- This project is best presented as an applied GenAI/RAG product: document ingestion, vector retrieval, prompt design, and LLM-powered workflow automation.
- For MLE or AI Engineer roles, the strongest signals are the retrieval pipeline, evaluation-oriented resume example dataset, and optional speech-interface prototype.
- TTS/STT was explored during the project and is included as an optional CLI prototype.
- Pinecone is loaded lazily, so the core app can still run without a Pinecone index unless example-resume retrieval is enabled.
- Keep all API keys and private resumes local.
