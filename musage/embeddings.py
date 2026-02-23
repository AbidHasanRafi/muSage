"""Lightweight embeddings and similarity search using sentence transformers"""

import pickle
import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    # Suppress noisy FAISS loader info messages
    import logging as _logging
    _logging.getLogger("faiss.loader").setLevel(_logging.WARNING)
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using fallback similarity search")

from sentence_transformers import SentenceTransformer

from . import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Lightweight embedding engine for semantic similarity
    Uses CPU-friendly all-MiniLM-L6-v2 model
    """

    def __init__(self):
        logger.info("Loading embedding model (this may take a moment on first run)...")
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device='cpu',
        )
        # Disable any progress bars the model might emit
        self.model.show_progress_bar = False
        logger.info("Embedding model loaded successfully")

        self.embeddings_cache = []
        self.texts_cache = []
        self.metadata_cache = []

        # FAISS index for fast similarity search
        self.index = None
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2

        self._load_cache()

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    def add_to_index(self, text: str, metadata: dict = None):
        """Add text to the searchable index"""
        embedding = self.embed(text)
        self.embeddings_cache.append(embedding)
        self.texts_cache.append(text)
        self.metadata_cache.append(metadata or {})

        # Rebuild index
        self._rebuild_index()
        self._save_cache()

        logger.debug(f"Added to index: {text[:50]}...")

    def add_batch_to_index(self, texts: List[str], metadatas: List[dict] = None):
        """Add multiple texts to index"""
        if not texts:
            return

        embeddings = self.embed_batch(texts)
        self.embeddings_cache.extend(embeddings)
        self.texts_cache.extend(texts)

        if metadatas:
            self.metadata_cache.extend(metadatas)
        else:
            self.metadata_cache.extend([{}] * len(texts))

        self._rebuild_index()
        self._save_cache()

        logger.info(f"Added {len(texts)} items to index")

    def _rebuild_index(self):
        """Rebuild FAISS index"""
        if not self.embeddings_cache:
            return

        embeddings_array = np.array(self.embeddings_cache).astype('float32')

        if FAISS_AVAILABLE:
            # Use FAISS for efficient search
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
        else:
            # Fallback: store normalized embeddings
            self.index = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar texts
        Returns: List of (text, similarity_score, metadata) tuples
        """
        if not self.embeddings_cache:
            return []

        query_embedding = self.embed(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        if FAISS_AVAILABLE and self.index:
            # FAISS search
            faiss.normalize_L2(query_embedding)
            k = min(k, len(self.texts_cache))
            similarities, indices = self.index.search(query_embedding, k)

            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx < len(self.texts_cache):  # Safety check
                    results.append((
                        self.texts_cache[idx],
                        float(similarity),
                        self.metadata_cache[idx]
                    ))
            return results

        else:
            # Fallback: NumPy similarity search
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(self.index, query_norm.T).flatten()

            # Get top k
            k = min(k, len(self.texts_cache))
            top_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_indices:
                results.append((
                    self.texts_cache[idx],
                    float(similarities[idx]),
                    self.metadata_cache[idx]
                ))
            return results

    def _save_cache(self):
        """Save embeddings cache to disk"""
        try:
            cache_data = {
                "embeddings": self.embeddings_cache,
                "texts": self.texts_cache,
                "metadata": self.metadata_cache,
            }
            with open(config.EMBEDDING_CACHE_FILE, "wb") as f:
                pickle.dump(cache_data, f)
            logger.debug("Embeddings cache saved")
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")

    def _load_cache(self):
        """Load embeddings cache from disk"""
        try:
            if config.EMBEDDING_CACHE_FILE.exists():
                with open(config.EMBEDDING_CACHE_FILE, "rb") as f:
                    cache_data = pickle.load(f)
                self.embeddings_cache = cache_data.get("embeddings", [])
                self.texts_cache = cache_data.get("texts", [])
                self.metadata_cache = cache_data.get("metadata", [])

                # Rebuild index
                if self.embeddings_cache:
                    self._rebuild_index()

                logger.info(f"Loaded {len(self.texts_cache)} items from embeddings cache")
        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {e}")

    def clear_cache(self):
        """Clear all cached embeddings"""
        self.embeddings_cache = []
        self.texts_cache = []
        self.metadata_cache = []
        self.index = None
        self._save_cache()
        logger.info("Embeddings cache cleared")

    def get_stats(self) -> dict:
        """Get statistics about the embedding index"""
        return {
            "total_embeddings": len(self.embeddings_cache),
            "dimension": self.dimension,
            "using_faiss": FAISS_AVAILABLE,
        }
