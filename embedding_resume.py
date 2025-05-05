import os
import json
from openai import OpenAI
from docx import Document
from dotenv import load_dotenv


client = OpenAI()

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("OpenAI API key not found. Please check your .env file.")
    exit(1)

def get_docx_content(file_path):
    """
    Extract text content from a .docx file.
    """
    try:
        doc = Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def generate_embedding(text):
    """
    Generate an embedding for a given text using OpenAI's embedding model.
    """
    try:
        # Call the OpenAI Embedding API
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        # Extract the embedding from the response
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_docx_files(directory_path):
    """
    Process all .docx files in a directory and generate embeddings.
    Returns a dictionary mapping file names to embeddings and metadata.
    """
    embeddings = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".docx"):  # Process only .docx files
            file_path = os.path.join(directory_path, file_name)
            print(f"Processing file: {file_path}")
            
            content = get_docx_content(file_path)
            if content:
                embedding = generate_embedding(content)
                if embedding:
                    embeddings[file_name] = {
                        "embedding": embedding,
                        "content": content,
                        "file_path": file_path
                    }
    return embeddings

# Example usage
if __name__ == "__main__":
    # Path to the directory containing .docx files
    directory = "./Resume_Text_Preprocessing"

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        exit(1)

    # Process .docx files and generate embeddings
    embeddings = process_docx_files(directory)

    # Save the embeddings with metadata to a JSON file
    output_file = "embeddings_with_metadata.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f)
        print(f"Embeddings with metadata saved to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings to file: {e}")