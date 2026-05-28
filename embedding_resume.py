import argparse
import json
import os
from pathlib import Path

from docx import Document
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_INPUT_DIR = "Resume_Text_Preprocessing"
DEFAULT_OUTPUT_FILE = "embeddings_with_metadata.json"
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_docx_content(file_path: Path) -> str | None:
    """Extract text content from a .docx file."""
    try:
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    except Exception as exc:
        print(f"Error reading file {file_path}: {exc}")
        return None


def generate_embedding(client: OpenAI, text: str, model: str) -> list[float] | None:
    """Generate an embedding for a text block."""
    try:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as exc:
        print(f"Error generating embedding: {exc}")
        return None


def process_docx_files(directory_path: Path, client: OpenAI, model: str) -> dict:
    """Process all .docx files in a directory and generate embedding metadata."""
    embeddings = {}
    for file_path in sorted(directory_path.glob("*.docx")):
        print(f"Processing file: {file_path}")

        content = get_docx_content(file_path)
        if not content:
            continue

        embedding = generate_embedding(client, content, model)
        if embedding:
            embeddings[file_path.name] = {
                "embedding": embedding,
                "content": content,
                "file_path": str(file_path),
            }

    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resume example embeddings from .docx files.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing .docx resumes.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSON path.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="OpenAI embedding model.")
    return parser.parse_args()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to your local .env file.")

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    client = OpenAI()
    embeddings = process_docx_files(input_dir, client, args.model)

    output_file.write_text(json.dumps(embeddings, indent=2), encoding="utf-8")
    print(f"Saved {len(embeddings)} embeddings to {output_file}")


if __name__ == "__main__":
    main()
