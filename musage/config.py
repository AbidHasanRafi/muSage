"""Configuration module for μSage (MuSage)"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path.home() / ".musage"
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CACHE_DIR = DATA_DIR / "cache"
CONVERSATION_DIR = DATA_DIR / "conversations"

# Create directories if they don't exist
for directory in [BASE_DIR, DATA_DIR, KNOWLEDGE_DIR, CACHE_DIR, CONVERSATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Web scraping settings
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Search settings
MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT = 15

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight CPU-friendly model
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.pkl"
KNOWLEDGE_BASE_FILE = KNOWLEDGE_DIR / "knowledge_base.pkl"
FAISS_INDEX_FILE = CACHE_DIR / "faiss_index.bin"

# Conversation settings
MAX_CONVERSATION_HISTORY = 10
CONVERSATION_FILE = CONVERSATION_DIR / "current_session.json"

# Response generation settings
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to use cached knowledge
MAX_SCRAPE_LENGTH = 5000  # Max characters to extract from a webpage

# Local LLM settings
USE_LOCAL_LLM = True  # Enable local LLM for answer generation (downloads model on first run)
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Lightweight model (~500MB, fast!)
# Alternative models (larger = better quality but slower):
#   "Qwen/Qwen2.5-1.5B-Instruct" - 1.5B params, ~3GB (better quality)
#   "microsoft/Phi-3-mini-4k-instruct" - 3.8B params, ~7GB (best quality)
#   "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - 1.1B params, ~1GB (good balance)
LLM_MAX_TOKENS = 300  # Maximum tokens to generate per response
LLM_CACHE_DIR = BASE_DIR / "models"  # Where to store downloaded model weights

# Create LLM cache directory
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Offline mode
OFFLINE_MODE = False  # Will be detected automatically

# CLI settings
CLI_PROMPT = "You"
CLI_ASSISTANT = "μSage"
CLI_WIDTH = 80
