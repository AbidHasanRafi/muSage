"""Knowledge base module for storing and retrieving learned information"""

import pickle
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


class KnowledgeEntry:
    """Single knowledge entry"""

    def __init__(self, query: str, content: str, source: str, metadata: Dict = None):
        self.query = query
        self.content = content
        self.source = source
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.access_count = 0
        self.last_accessed = None
        self.usefulness_score = 0.5  # Neutral start

    def access(self):
        """Track access"""
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

    def rate_usefulness(self, score: float):
        """Update usefulness score (0-1)"""
        # Moving average
        self.usefulness_score = 0.7 * self.usefulness_score + 0.3 * score

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "usefulness_score": self.usefulness_score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeEntry":
        """Create from dictionary"""
        entry = cls(
            query=data["query"],
            content=data["content"],
            source=data["source"],
            metadata=data.get("metadata", {}),
        )
        entry.created_at = data.get("created_at", entry.created_at)
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed")
        entry.usefulness_score = data.get("usefulness_score", 0.5)
        return entry


class KnowledgeBase:
    """
    Local knowledge storage system
    Stores learned information with metadata
    """

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or config.KNOWLEDGE_BASE_FILE
        self.entries: List[KnowledgeEntry] = []
        self.load()

    def add_entry(self, query: str, content: str, source: str, metadata: Dict = None):
        """Add a new knowledge entry"""
        entry = KnowledgeEntry(query, content, source, metadata)
        self.entries.append(entry)
        logger.info(f"Added knowledge entry for: {query[:50]}...")
        self.save()

    def get_all_entries(self) -> List[KnowledgeEntry]:
        """Get all knowledge entries"""
        return self.entries

    def get_entry_by_index(self, index: int) -> Optional[KnowledgeEntry]:
        """Get entry by index"""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def mark_useful(self, index: int, useful: bool = True):
        """Mark an entry as useful or not"""
        entry = self.get_entry_by_index(index)
        if entry:
            score = 1.0 if useful else 0.0
            entry.rate_usefulness(score)
            self.save()

    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        if not self.entries:
            return {
                "total_entries": 0,
                "avg_usefulness": 0,
                "most_accessed": None,
            }

        return {
            "total_entries": len(self.entries),
            "avg_usefulness": sum(e.usefulness_score for e in self.entries) / len(self.entries),
            "most_accessed": max(self.entries, key=lambda e: e.access_count).query if self.entries else None,
        }

    def save(self):
        """Save knowledge base to disk"""
        try:
            data = [entry.to_dict() for entry in self.entries]
            with open(self.storage_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {len(self.entries)} entries to knowledge base")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def load(self):
        """Load knowledge base from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                    self.entries = [KnowledgeEntry.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.entries)} entries from knowledge base")
            else:
                logger.info("No existing knowledge base found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            self.entries = []

    def clear(self):
        """Clear all entries (use with caution!)"""
        self.entries = []
        self.save()
        logger.info("Knowledge base cleared")
