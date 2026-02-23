"""Main entry point for μSage (MuSage) when run as a module"""

# ── Must run BEFORE any third-party imports ─────────────────────────────────
import logging
import sys
import os
import warnings

# Force UTF-8 output on Windows (fixes garbled box/block characters)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Environment flags (must be set before torch/transformers load) ───────────
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')   # kill "unauthenticated" banner
os.environ.setdefault('TRANSFORMERS_VERBOSITY',        'error') # silence transformers verbosity
os.environ.setdefault('TOKENIZERS_PARALLELISM',        'false') # silence tokenizer fork warning
os.environ.setdefault('TQDM_DISABLE',                  '1')     # disable all tqdm progress bars

# ── Python warnings filter ───────────────────────────────────────────────────
warnings.filterwarnings('ignore')   # suppress any UserWarning/FutureWarning from libraries

# ── Root logger ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# ── Silence specific verbose libraries ──────────────────────────────────────
for _noisy in (
    'httpx', 'httpcore',
    'sentence_transformers', 'sentence_transformers.SentenceTransformer',
    'transformers', 'transformers.modeling_utils', 'transformers.configuration_utils',
    'transformers.tokenization_utils_base', 'transformers.utils.hub',
    'huggingface_hub', 'huggingface_hub.utils._http', 'huggingface_hub.file_download',
    'huggingface_hub.repocard', 'huggingface_hub._commit_api',
    'primp', 'faiss', 'faiss.loader', 'urllib3',
    'musage.knowledge', 'musage.embeddings', 'musage.scraper',
    'musage.search', 'musage.agent', 'musage.conversation',
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from musage.cli import main

if __name__ == '__main__':
    main()
