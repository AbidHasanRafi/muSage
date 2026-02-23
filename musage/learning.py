"""
Continuous Learning System for μSage
Learns from user interactions and improves over time.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# Storage paths
LEARNING_DIR = Path.home() / ".musage" / "learning"
LEARNED_QA_FILE = LEARNING_DIR / "learned_qa.json"
USAGE_LOG_FILE = LEARNING_DIR / "usage_log.json"
FEEDBACK_FILE = LEARNING_DIR / "feedback.json"
STATS_FILE = LEARNING_DIR / "stats.json"

# Ensure directories exist
LEARNING_DIR.mkdir(parents=True, exist_ok=True)


class LearningSystem:
    """Tracks usage, collects feedback, and learns new Q&A pairs"""
    
    def __init__(self):
        self.learned_qa = self._load_learned_qa()
        self.usage_stats = self._load_stats()
        self.session_queries = []
    
    def _load_learned_qa(self) -> Dict[str, dict]:
        """Load previously learned Q&A pairs"""
        if LEARNED_QA_FILE.exists():
            try:
                with open(LEARNED_QA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_learned_qa(self):
        """Save learned Q&A pairs to disk"""
        with open(LEARNED_QA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.learned_qa, f, indent=2, ensure_ascii=False)
    
    def _load_stats(self) -> Dict:
        """Load usage statistics"""
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._default_stats()
        return self._default_stats()
    
    def _default_stats(self) -> Dict:
        """Default statistics structure"""
        return {
            'total_queries': 0,
            'learned_qa_count': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'top_topics': {},
            'first_used': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
        }
    
    def _save_stats(self):
        """Save statistics to disk"""
        self.usage_stats['last_used'] = datetime.now().isoformat()
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.usage_stats, f, indent=2)
    
    def log_query(self, query: str, answer: str, method: str, success: bool):
        """
        Log a query and its result
        
        Args:
            query: User's question
            answer: Generated answer
            method: Method used (simple_qa, minigpt, llm, extraction)
            success: Whether answer was successfully generated
        """
        self.session_queries.append(query)
        
        # Update stats
        self.usage_stats['total_queries'] += 1
        
        # Track topic frequency
        topic = self._extract_topic(query)
        if topic:
            topics = self.usage_stats.get('top_topics', {})
            topics[topic] = topics.get(topic, 0) + 1
            self.usage_stats['top_topics'] = topics
        
        # Log to file
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer_length': len(answer) if answer else 0,
            'method': method,
            'success': success,
            'topic': topic,
        }
        
        self._append_to_log(log_entry)
        self._save_stats()
    
    def record_feedback(self, query: str, answer: str, helpful: bool, comment: str = ""):
        """
        Record user feedback on an answer
        
        Args:
            query: The question asked
            answer: The answer given
            helpful: Whether the answer was helpful
            comment: Optional user comment
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer[:200],  # First 200 chars
            'helpful': helpful,
            'comment': comment,
        }
        
        # Update stats
        if helpful:
            self.usage_stats['positive_feedback'] += 1
            
            # If helpful and used extraction/web, learn this Q&A
            if len(answer) > 50 and len(answer) < 500:
                self._learn_new_qa(query, answer)
        else:
            self.usage_stats['negative_feedback'] += 1
            
            # Handle negative feedback intelligently
            normalized_query = self._normalize_for_learning(query)
            
            # If this bad answer is in learned Q&A, remove it or reduce confidence
            if normalized_query in self.learned_qa:
                logger.info(f"Removing bad answer from learned Q&A: {normalized_query}")
                # Option 1: Remove completely
                del self.learned_qa[normalized_query]
                self._save_learned_qa()
                self.usage_stats['learned_qa_count'] = len(self.learned_qa)
            
            # Try to extract correction from user comment
            if comment:
                corrected_answer = self._extract_correction_from_comment(query, comment)
                if corrected_answer:
                    logger.info(f"Learning corrected answer from user feedback")
                    self._learn_new_qa(query, corrected_answer)
        
        # Save feedback
        self._append_to_feedback(feedback_entry)
        self._save_stats()
    
    def _learn_new_qa(self, query: str, answer: str):
        """
        Learn a new Q&A pair from successful interaction
        
        Args:
            query: The question
            answer: The good answer
        """
        normalized_query = self._normalize_for_learning(query)
        
        # Don't learn if too similar to existing
        if normalized_query in self.learned_qa:
            # Update confidence/usage count
            self.learned_qa[normalized_query]['usage_count'] += 1
            self.learned_qa[normalized_query]['last_used'] = datetime.now().isoformat()
        else:
            # Learn new Q&A
            self.learned_qa[normalized_query] = {
                'answer': answer,
                'learned_date': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'usage_count': 1,
                'confidence': 0.8,  # Initial confidence
            }
            
            self.usage_stats['learned_qa_count'] = len(self.learned_qa)
        
        self._save_learned_qa()
        self._save_stats()
    
    def get_learned_answer(self, query: str, threshold: float = 0.75) -> Optional[str]:
        """
        Try to get answer from learned Q&A
        
        Args:
            query: User's question
            threshold: Confidence threshold
            
        Returns:
            Answer string or None
        """
        normalized_query = self._normalize_for_learning(query)
        logger.info(f"[LEARNED Q&A DEBUG] Query: '{query}' → Normalized: '{normalized_query}'")
        logger.info(f"[LEARNED Q&A DEBUG] Database has {len(self.learned_qa)} entries, threshold={threshold}")
        
        # Exact match
        if normalized_query in self.learned_qa:
            entry = self.learned_qa[normalized_query]
            logger.info(f"[LEARNED Q&A DEBUG] Exact match found! Confidence: {entry['confidence']}")
            if entry['confidence'] >= threshold:
                # Update usage
                entry['usage_count'] += 1
                entry['last_used'] = datetime.now().isoformat()
                self._save_learned_qa()
                logger.info(f"[LEARNED Q& A DEBUG] ✅ Returning: {entry['answer'][:50]}...")
                return entry['answer']
            else:
                logger.info(f"[LEARNED Q&A DEBUG] ❌ Confidence too low ({entry['confidence']} < {threshold})")
        else:
            logger.info(f"[LEARNED Q&A DEBUG] ❌ No exact match, trying fuzzy...")
        
        # Fuzzy match (implemented later for efficiency)
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0
        
        for key, entry in self.learned_qa.items():
            if entry['confidence'] < threshold:
                continue
            
            score = SequenceMatcher(None, normalized_query, key).ratio()
            if score > best_score:
                best_score = score
                best_match = entry
        
        if best_score >= 0.85:  # High similarity required for learned Q&A
            return best_match['answer']
        
        return None
    
    def _normalize_for_learning(self, query: str) -> str:
        """Normalize query for learning storage"""
        query = query.lower().strip()
        query = re.sub(r'\?+$', '', query)
        query = re.sub(r'^(what is|what are|tell me about|explain)\s+(the|a|an)?\s*', '', query)
        query = query.strip()
        
        # Remove duplicate consecutive words (e.g., "full form form of" → "full form of")
        words = query.split()
        normalized_words = []
        prev_word = None
        for word in words:
            if word != prev_word:
                normalized_words.append(word)
                prev_word = word
        
        return ' '.join(normalized_words)
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract main topic from query"""
        # Remove question words
        topic = re.sub(r'^(what|how|why|when|where|who|which|tell me|explain)\s+', '', query.lower())
        topic = re.sub(r'^(is|are|was|were|the|a|an)\s+', '', topic)
        topic = re.sub(r'\?+$', '', topic)
        
        # Take first 3 words max
        words = topic.strip().split()[:3]
        return ' '.join(words) if words else None
    
    def _extract_correction_from_comment(self, query: str, comment: str) -> Optional[str]:
        """
        Extract corrected answer from user's feedback comment
        
        Patterns recognized:
        - "The X is Y"
        - "It should be Y"
        - "Correct answer is Y"
        - "Actually it's Y"
        - "X means Y"
        
        Args:
            query: Original query
            comment: User's comment with potential correction
            
        Returns:
            Corrected answer or None
        """
        import re
        
        # Pattern 1: "The X is/are Y" (most specific)
        match = re.search(r'[Tt]he\s+.{1,50}\s+(?:is|are)\s+([^,\.!?]+)', comment)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 5 and len(correction) < 200:
                logger.info(f"Extracted correction (pattern 1): {correction}")
                return correction
        
        # Pattern 2: "correct/right/actual answer is Y"
        match = re.search(r'(?:correct|right|actual)\s+answer\s+(?:is|are)\s+([^,\.!?]+)', comment, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 2 and len(correction) < 200:
                logger.info(f"Extracted correction (pattern 2): {correction}")
                return correction
        
        # Pattern 3: "it/they should be Y" or "it/they is/are Y"
        match = re.search(r'(?:it|they)\s+(?:should be|is|are)\s+([^,\.!?]+)', comment, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 5 and len(correction) < 200:
                logger.info(f"Extracted correction (pattern 3): {correction}")
                return correction
        
        # Pattern 4: "actually/actually it's Y"
        match = re.search(r'[Aa]ctually(?:\s+it\'?s?)?\s+([^,\.!?]+)', comment)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 5 and len(correction) < 200:
                logger.info(f"Extracted correction (pattern 4): {correction}")
                return correction
        
        # Pattern 5: "X means/stands for Y"
        match = re.search(r'(?:means|stands for)\s+([^,\.!?]+)', comment, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 3 and len(correction) < 200:
                logger.info(f"Extracted correction (pattern 5): {correction}")
                return correction
        
        # Pattern 6: "should be Y" (less specific)
        match = re.search(r'should\s+be\s+([^,\.!?]+)', comment, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 5 and len(correction) < 200:
                # Avoid false positives like "should be more detailed"
                if not any(word in correction.lower() for word in ['more', 'better', 'clearer', 'detailed', 'specific']):
                    logger.info(f"Extracted correction (pattern 6): {correction}")
                    return correction
        
        # Pattern 7: Simple "is Y" at the start (after filtering)
        match = re.search(r'^[^,\.!?]+\s+(?:is|are)\s+([^,\.!?]+)', comment, re.IGNORECASE)
        if match:
            correction = match.group(1).strip()
            if len(correction) > 3 and len(correction) < 200:
                logger.info(f"Extracted correction (pattern 7): {correction}")
                return correction
        
        return None
    
    def _append_to_log(self, entry: dict):
        """Append entry to usage log"""
        logs = []
        if USAGE_LOG_FILE.exists():
            try:
                with open(USAGE_LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        logs.append(entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(USAGE_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
    
    def _append_to_feedback(self, entry: dict):
        """Append feedback entry"""
        feedback = []
        if FEEDBACK_FILE.exists():
            try:
                with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                    feedback = json.load(f)
            except:
                feedback = []
        
        feedback.append(entry)
        
        # Keep all feedback (it's valuable!)
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2)
    
    def get_stats_summary(self) -> str:
        """Get formatted statistics summary"""
        stats = self.usage_stats
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           μSage Learning Statistics                          ║
╚══════════════════════════════════════════════════════════════╝

📊 Usage:
   • Total queries: {stats['total_queries']}
   • Learned Q&A pairs: {stats['learned_qa_count']}
   • First used: {stats.get('first_used', 'N/A')[:10]}
   • Last used: {stats.get('last_used', 'N/A')[:10]}

👍 Feedback:
   • Positive: {stats['positive_feedback']}
   • Negative: {stats['negative_feedback']}
   • Satisfaction: {self._calc_satisfaction()}%

🔥 Top Topics:
"""
        
        # Add top 5 topics
        top_topics = sorted(
            stats.get('top_topics', {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for topic, count in top_topics:
            summary += f"   • {topic}: {count} queries\n"
        
        if not top_topics:
            summary += "   • (no topics tracked yet)\n"
        
        summary += "\n💡 The more you use μSage, the smarter it gets!"
        
        return summary
    
    def _calc_satisfaction(self) -> int:
        """Calculate satisfaction percentage"""
        total = self.usage_stats['positive_feedback'] + self.usage_stats['negative_feedback']
        if total == 0:
            return 0
        return int((self.usage_stats['positive_feedback'] / total) * 100)
    
    def export_learned_qa(self, output_file: str):
        """
        Export learned Q&A to a Python file for integration
        
        Args:
            output_file: Path to output Python file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('# Auto-generated learned Q&A pairs\n')
            f.write('# Add these to musage/simple_qa.py QA_DATABASE\n\n')
            f.write('LEARNED_QA = {\n')
            
            for query, data in sorted(self.learned_qa.items()):
                if data['confidence'] >= 0.8:  # Only export high-confidence
                    answer = data['answer'].replace('"', '\\"')
                    f.write(f'    "{query}": "{answer}",\n')
            
            f.write('}\n')
        
        print(f"✅ Exported {len(self.learned_qa)} learned Q&A pairs to {output_file}")
    
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on usage patterns"""
        suggestions = []
        
        # Analyze negative feedback
        if self.usage_stats['negative_feedback'] > 5:
            suggestions.append("💡 Consider reviewing questions with negative feedback in feedback.json")
        
        # Check if learning is working
        if self.usage_stats['total_queries'] > 50 and self.usage_stats['learned_qa_count'] < 5:
            suggestions.append("💡 Try marking more answers as helpful to build your learned Q&A database")
        
        # Check topic coverage
        top_topics = self.usage_stats.get('top_topics', {})
        if len(top_topics) > 10:
            most_common = max(top_topics.items(), key=lambda x: x[1])
            if most_common[1] > 10:
                suggestions.append(f"💡 You ask a lot about '{most_common[0]}' - consider adding more Q&A in this domain")
        
        return suggestions


# Singleton instance
_learning_system = None

def get_learning_system() -> LearningSystem:
    """Get or create the learning system singleton"""
    global _learning_system
    if _learning_system is None:
        _learning_system = LearningSystem()
    return _learning_system
