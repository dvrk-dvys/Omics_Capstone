import os
import threading

# Prevent HuggingFace tokenizers from forking child processes when called
# from multiple threads — avoids SIGSEGV on macOS with ThreadPoolExecutor.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED = None
_ENCODE_LOCK = threading.Lock()


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
        EMBED = SentenceTransformer(MODEL_NAME, device=EMBED_DEVICE)
    return EMBED

def encode_safe(texts, **kwargs):
    """Thread-safe wrapper around model.encode() — serializes calls via lock."""
    with _ENCODE_LOCK:
        return get_model().encode(texts, **kwargs)