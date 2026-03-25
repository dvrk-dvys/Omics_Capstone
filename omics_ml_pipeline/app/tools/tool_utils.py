import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED = None


def chunk_text(document, max_tokens=1000, overlap=200):
    """
    Simple chunker: split by headings/blank lines then merge small parts.
    # target chunk size in characters (or tokens)
    # overlap to maintain context between chunks
    # The splitter will try to split by double newline (paragraph), then newline, then space, then as last resort character.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )
    chunks = text_splitter.split_text(document)
    return chunks

def get_model():
    global EMBED
    if EMBED is None:
        # Use MPS on Apple Silicon; falls back to CPU if unavailable
        # device = "mps" if SentenceTransformer(MODEL_NAME).device.type != "cuda" else "cuda"
        EMBED = SentenceTransformer(MODEL_NAME, device=EMBED_DEVICE)
    return EMBED