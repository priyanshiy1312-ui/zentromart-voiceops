"""Central configuration. Everything is overridable through environment variables / .env."""
import os
from datetime import date
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent

# Where the extracted ZentroMart dataset lives (NOT copied into the repo).
DATASET_DIR = Path(os.getenv("DATASET_DIR", "Capstone_Project_Dataset_FTTT_2026")).expanduser()
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"

# The dataset's reference "today" (brief: treat 20 October 2026 as today's date).
TODAY = date.fromisoformat(os.getenv("TODAY", "2026-10-20"))

# ---- LLM (any provider: anthropic, or any OpenAI-compatible endpoint) ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()  # anthropic | openai
_DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "claude-sonnet-5-5"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # e.g. Groq / Gemini / Ollama endpoints

# ---- Retrieval ----
# "none" -> BM25 only (no model download). Otherwise a sentence-transformers model name.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
TOP_K = int(os.getenv("TOP_K", "6"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(ROOT / ".cache")))

# ---- Gnani speech (only STT/TTS vendor allowed by the rules) ----
GNANI_TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_VOICE = os.getenv("GNANI_VOICE", "Kaveri")
DEFAULT_LANG = os.getenv("VOICE_LANG", "en-IN")
