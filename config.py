import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# --- ArangoDB ---
ARANGO_URL: str = os.getenv("ARANGO_URL", "http://localhost:8529")
ARANGO_DB: str = os.getenv("ARANGO_DB", "graphrag")
ARANGO_USER: str = os.getenv("ARANGO_USER", "root")
ARANGO_PASSWORD: str = os.getenv("ARANGO_PASSWORD", "")

# --- Collections ---
COL_DOCUMENTS = "documents"
COL_CHUNKS = "chunks"
COL_ENTITIES = "entities"
EDGE_RELATIONS = "relationships"
EDGE_MENTIONS = "mentions"
EDGE_BELONGS = "belongs_to"
GRAPH_NAME = "knowledge_graph"

# --- Chunking ---
CHUNK_SIZE = 500          # tokens (approx chars / 4)
CHUNK_OVERLAP = 60

# --- Embeddings ---
EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5

# --- LLMs ---
GROQ_EXTRACT_MODEL = "llama-3.1-8b-instant"     # fast, for entity extraction
GROQ_ANSWER_MODEL = "llama-3.3-70b-versatile"   # powerful, for final answer

# --- Retrieval ---
TOP_K_CHUNKS = 6
TOP_K_ENTITIES = 10
GRAPH_HOP_DEPTH = 2

# --- Rate limits (free tier conservative) ---
GROQ_RPM = 25           # requests per minute (stay under 30)
GEMINI_EMBED_RPM = 1400  # well under 1500
