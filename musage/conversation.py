"""Conversation memory manager"""

import json
import logging
from typing import List, Dict
from datetime import datetime

from . import config

logger = logging.getLogger(__name__)


class ConversationMessage:
    """Single conversation message"""

    def __init__(self, role: str, content: str, timestamp: str = None):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
        )


class ConversationManager:
    """
    Manages conversation history and context
    Provides memory for continuity
    """

    def __init__(self):
        self.history: List[ConversationMessage] = []
        self.session_start = datetime.now().isoformat()
        self.load()

    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        message = ConversationMessage(role, content)
        self.history.append(message)

        # Keep only recent history
        if len(self.history) > config.MAX_CONVERSATION_HISTORY * 2:
            self.history = self.history[-config.MAX_CONVERSATION_HISTORY * 2:]

        self.save()

    def get_recent_context(self, n: int = None) -> List[ConversationMessage]:
        """Get recent conversation messages"""
        if n is None:
            n = config.MAX_CONVERSATION_HISTORY
        return self.history[-n:]

    def get_context_string(self, n: int = 5) -> str:
        """Get recent context as a formatted string"""
        recent = self.get_recent_context(n)
        context_parts = []

        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")

        return "\n".join(context_parts)

    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.session_start = datetime.now().isoformat()
        self.save()
        logger.info("Conversation history cleared")

    def save(self):
        """Save conversation to disk"""
        try:
            data = {
                "session_start": self.session_start,
                "history": [msg.to_dict() for msg in self.history],
            }
            with open(config.CONVERSATION_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    def load(self):
        """Load conversation from disk"""
        try:
            if config.CONVERSATION_FILE.exists():
                with open(config.CONVERSATION_FILE, "r") as f:
                    data = json.load(f)
                    self.session_start = data.get("session_start", self.session_start)
                    self.history = [
                        ConversationMessage.from_dict(msg)
                        for msg in data.get("history", [])
                    ]
                logger.info(f"Loaded conversation with {len(self.history)} messages")
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")

    def get_summary(self) -> Dict:
        """Get conversation summary"""
        user_messages = sum(1 for msg in self.history if msg.role == "user")
        assistant_messages = sum(1 for msg in self.history if msg.role == "assistant")

        return {
            "total_messages": len(self.history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "session_start": self.session_start,
        }
