"""Response generation module — RAG-style coherent extraction"""

import logging
import re
from typing import List, Tuple, Optional, Dict, Set

from . import config
from .builtin_knowledge import get_polite_cant_answer

# Try to import LLM modules (optional)
try:
    from . import llm as local_llm
    LLM_AVAILABLE = True
except ImportError:
    local_llm = None
    LLM_AVAILABLE = False

# Import simple Q&A (zero dependencies, always available)
try:
    from . import simple_qa
    SIMPLE_QA_AVAILABLE = simple_qa.is_available()
except ImportError:
    simple_qa = None
    SIMPLE_QA_AVAILABLE = False

# Try to import MiniGPT (lightweight, zero-dependency alternative)
try:
    from . import minigpt
    MINIGPT_AVAILABLE = minigpt.is_available()
except ImportError:
    minigpt = None
    MINIGPT_AVAILABLE = False

# Import learning system (continuous improvement)
try:
    from .learning import get_learning_system
    LEARNING_AVAILABLE = True
except ImportError:
    get_learning_system = None
    LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Stop words (excluded from relevance scoring) ─────────────────────────────
_STOP: Set[str] = frozenset({
    'what','is','are','how','why','when','where','who','the','a','an','of',
    'to','in','and','or','for','do','does','i','my','can','you','it','its',
    'was','be','been','this','that','with','from','at','by','on','as','up',
    'about','into','than','then','so','if','but','not','no','we','they','he',
    'she','will','would','could','should','have','has','had','just','also',
    'very','more','most','some','any','all','one','two','three','get','make',
    'use','used','using','new','good','like','well','time','way','help','our',
    'their','these','those','been','which','there','here','each','an','its',
})

# ── Noise patterns — reject these sentences entirely ─────────────────────────
_NOISE_RE = re.compile(
    r'(cookie[s\s]+(policy|consent|notice)|privacy\s+policy|'
    r'subscribe|click\s+here|sign\s+up|log\s*in|advertisement|'
    r'sponsored|copyright\s+\d{4}|all\s+rights\s+reserved|'
    r'\d+\s*min(ute)?\s+read|share\s+(this|on)|follow\s+us|'
    r'newsletter|terms\s+of\s+(use|service)|'
    r'(read|see)\s+more\s+(about|on|at)|'
    r'(updated|published|posted)\s+on\s+\w+\s+\d+|'
    r'image\s+(source|caption)|'
    r'getty\s+images?|AFP|\bAP\s+Photo\b|PA\s+Media|'
    r'alamy|shutterstock|\bepa\b|'
    r'(watch|listen)\s+(the|this|full)|'
    r'related\s+(article|story|content|questions?)|'
    r'trending\s+questions?|continue\s+learning|search\s+continue|'
    r'by\s+\w+\s+\w+,?\s+(bbc|cnn|fox|nbc|sky)\s+(news|sport)?|'
    r'what\s+is\s+the\s+ratio|how\s+many\s+times|is\s+five\s+times)',
    re.I,
)

# ── Question patterns that shouldn't be in answers ───────────────────────────
_QUESTION_SENTENCE_RE = re.compile(
    r'^\s*(what|how|why|when|where|who|which|is\s+\w+\s+\w+\?)',
    re.I,
)

# ── Definition markers ────────────────────────────────────────────────────────
_DEF_RE = re.compile(
    r'\b(is\s+(a|an|the|one)\b|is\s+defined\s+as\b|refers?\s+to\b|'
    r'defined\s+as\b|describes?\s+\b|means?\s+\b|known\s+as\b|is\s+called\b|'
    r'is\s+(the\s+)?(study|branch|field|process|method|technique|'
    r'concept|practice|science|art|system|language|framework|tool|'
    r'approach|way|type|form|kind)\b|'
    r'can\s+be\s+defined|in\s+mathematics|in\s+(computer\s+)?science\b|'
    r'in\s+simple\s+terms|put\s+simply|essentially)',
    re.I,
)

# ── Example markers ───────────────────────────────────────────────────────────
_EXAMPLE_RE = re.compile(
    r'\b(for\s+example|for\s+instance|such\s+as|e\.g\.|i\.e\.|'
    r'to\s+illustrate|as\s+an\s+example|consider\s+(the\s+)?example|'
    r'like\s+\w+(\s+and\s+\w+)?|including\b)',
    re.I,
)

# ── Elaboration / key-fact markers ───────────────────────────────────────────
_ELABORATION_RE = re.compile(
    r'\b(allows?\b|enables?\b|provides?\b|consists?\s+of\b|'
    r'includes?\b|involves?\b|requires?\b|uses?\b|based\s+on\b|'
    r'composed\s+of\b|made\s+(up\s+)?of\b|results?\s+in\b|'
    r'key\s+(feature|aspect|advantage|property|concept)\b|'
    r'important\b|primary\b|main\b|fundamental\b|core\b)',
    re.I,
)

# ── Code request patterns ─────────────────────────────────────────────────────
_CODE_REQUEST_RE = re.compile(
    r'\b(write|create|make|generate|give\s+me|show\s+me)\s+('
    r'(a|an|some|the)\s+)?(code|program|script|function|class|snippet)|'
    r'\b(write|create|make|generate|give|show)\s+.*?\s*(python|javascript|java|c\+\+|html|css)\s+(code|program|script|function)|'
    r'(python|javascript|java|c\+\+|html|css)\s+code\s+(to|for|that)|'
    r'code\s+(to|for|that)\s+(print|display|sort|calculate|find)',
    re.I,
)


def _clean(s: str) -> str:
    """Remove URLs, footnote markers, excess whitespace from a sentence."""
    s = re.sub(r'https?://\S+', '', s)
    s = re.sub(r'\[\d+\]', '', s)          # [1], [2] reference markers
    s = re.sub(r'\s+', ' ', s).strip()
    if s and not s[0].isupper():
        s = s[0].upper() + s[1:]
    return s


def _is_good_sentence(s: str) -> bool:
    """Return True if the sentence is content-bearing and readable."""
    s = s.strip()
    if len(s) < 35:
        return False
    words = s.split()
    if len(words) < 6:
        return False
    # Reject noise
    if _NOISE_RE.search(s):
        return False
    # Reject question sentences (they're not answers)
    if _QUESTION_SENTENCE_RE.match(s):
        return False
    # Reject slash-heavy lines (table rows, breadcrumbs)
    if s.count('/') + s.count('|') + s.count('\\') > 3:
        return False
    # Reject mostly non-alpha (code snippets, URLs, data tables)
    alpha = sum(1 for c in s if c.isalpha())
    if alpha < len(s) * 0.45:
        return False
    # Reject lines that look like headings (short, no verb)
    if len(words) <= 4 and not re.search(r'\b(is|are|was|were|has|have|can|do|does)\b', s, re.I):
        return False
    # Reject sentences with excessive numbers/symbols (likely table data or list items)
    digit_ratio = sum(1 for c in s if c.isdigit()) / len(s)
    if digit_ratio > 0.20:
        return False
    return True


def _sentence_terms(s: str) -> Set[str]:
    return set(re.findall(r'\w+', s.lower())) - _STOP


def _split_sentences(text: str) -> List[str]:
    """Split text into cleaned, filtered sentences."""
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    out = []
    for s in raw:
        s = _clean(s.strip())
        if _is_good_sentence(s):
            out.append(s)
    return out


def _is_mostly_english(text: str) -> bool:
    """
    Return True if text is predominantly English (Latin alphabet).
    Rejects responses in Chinese, Japanese, Arabic, Cyrillic, etc.
    """
    if not text or len(text.strip()) < 20:
        return True  # too short to judge
    
    # Count latin letters vs total non-whitespace chars
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 0x0370)
    total = sum(1 for c in text if not c.isspace())
    
    if total == 0:
        return True
    
    latin_ratio = latin / total
    # If less than 50% latin alphabet, likely wrong language
    return latin_ratio >= 0.50


def _response_relevance(query: str, response: str) -> float:
    """
    Calculate how relevant the response is to the query (0.0-1.0).
    Returns very low score if response seems completely unrelated.
    """
    query_terms = set(re.findall(r'\w+', query.lower())) - _STOP
    response_terms = set(re.findall(r'\w+', response.lower())) - _STOP
    
    if not query_terms:
        return 1.0
    
    overlap = len(query_terms & response_terms)
    return overlap / len(query_terms)


# ═════════════════════════════════════════════════════════════════════════════
# § DISCRIMINATOR — Quality Evaluator (Generator-Discriminator paradigm)
# ═════════════════════════════════════════════════════════════════════════════

class ResponseDiscriminator:
    """
    Evaluates response quality across multiple dimensions.
    Part of the generator-discriminator refinement loop.
    """

    def evaluate(self, query: str, response: str, intent: str = 'general') -> Dict:
        """
        Return quality assessment:
        {
            'coherence': 0.0-1.0,      # logical flow, transitions
            'completeness': 0.0-1.0,   # query terms addressed
            'clarity': 0.0-1.0,        # no ambiguous references
            'conciseness': 0.0-1.0,    # no redundancy
            'structure': 0.0-1.0,      # proper format for intent
            'overall': 0.0-1.0,
            'issues': [str],           # specific problems found
        }
        """
        issues = []
        
        # ── 1. Coherence (logical flow, transition words) ─────────────────────
        coherence = self._check_coherence(response, issues)
        
        # ── 2. Completeness (query terms present) ────────────────────────────
        completeness = self._check_completeness(query, response, issues)
        
        # ── 3. Clarity (no orphaned pronouns) ────────────────────────────────
        clarity = self._check_clarity(response, issues)
        
        # ── 4. Conciseness (no repetition) ───────────────────────────────────
        conciseness = self._check_conciseness(response, issues)
        
        # ── 5. Structure (format matches intent) ─────────────────────────────
        structure = self._check_structure(query, response, intent, issues)
        
        # ── Overall score (weighted average) ─────────────────────────────────
        overall = (
            coherence * 0.2 + completeness * 0.3 + clarity * 0.25 +
            conciseness * 0.15 + structure * 0.1
        )
        
        return {
            'coherence': coherence,
            'completeness': completeness,
            'clarity': clarity,
            'conciseness': conciseness,
            'structure': structure,
            'overall': overall,
            'issues': issues,
        }
    
    def _check_coherence(self, response: str, issues: List[str]) -> float:
        """Check for logical flow and transitions between sentences."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words/phrases
        transitions = re.compile(
            r'\b(however|therefore|thus|moreover|furthermore|additionally|'
            r'for example|for instance|in addition|as a result|consequently|'
            r'similarly|likewise|on the other hand|in contrast|nevertheless|'
            r'meanwhile|subsequently|finally|first|second|third)\b',
            re.I
        )
        
        transition_count = sum(1 for s in sentences if transitions.search(s))
        transition_ratio = transition_count / max(len(sentences) - 1, 1)
        
        # Penalize if very few transitions in multi-sentence response
        if len(sentences) >= 4 and transition_ratio < 0.2:
            issues.append('lacks_transitions')
        
        # Check for abrupt topic shifts (significant word overlap between adjacent sentences)
        coherence_score = 0.7  # baseline
        if transition_ratio > 0.3:
            coherence_score = 0.95
        elif transition_ratio > 0.15:
            coherence_score = 0.85
        
        return coherence_score
    
    def _check_completeness(self, query: str, response: str, issues: List[str]) -> float:
        """Check if key query terms are addressed in response."""
        query_terms = set(re.findall(r'\w+', query.lower())) - _STOP
        response_terms = set(re.findall(r'\w+', response.lower())) - _STOP
        
        if not query_terms:
            return 1.0
        
        overlap = len(query_terms & response_terms)
        coverage = overlap / len(query_terms)
        
        if coverage < 0.3:
            issues.append('missing_query_terms')
            return 0.4
        elif coverage < 0.5:
            issues.append('partial_coverage')
            return 0.65
        
        return min(0.95, 0.5 + coverage * 0.6)
    
    def _check_clarity(self, response: str, issues: List[str]) -> float:
        """Check for ambiguous pronouns without clear antecedents."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if not sentences:
            return 1.0
        
        # Check first sentence for orphaned pronouns (no prior subject to refer to)
        first = sentences[0].lower()
        orphaned_pronouns = re.compile(r'\b(it|they|them|their|its|this|that|these|those)\b', re.I)
        
        if orphaned_pronouns.search(first):
            issues.append('orphaned_pronoun_in_first_sentence')
            return 0.6
        
        # Check for excessive pronoun use without clear subjects nearby
        pronoun_density = len(orphaned_pronouns.findall(response)) / max(len(response.split()), 1)
        if pronoun_density > 0.15:
            issues.append('excessive_pronouns')
            return 0.7
        
        return 0.95
    
    def _check_conciseness(self, response: str, issues: List[str]) -> float:
        """Check for repetitive content or redundant sentences."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s) > 20]
        if len(sentences) < 2:
            return 1.0
        
        # Check bigram overlap between sentences (indicates repetition)
        redundancy_count = 0
        for i, s1 in enumerate(sentences):
            words1 = set(re.findall(r'\w+', s1.lower())) - _STOP
            for s2 in sentences[i + 1:]:
                words2 = set(re.findall(r'\w+', s2.lower())) - _STOP
                if words1 and words2:
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    if overlap > 0.7:
                        redundancy_count += 1
        
        if redundancy_count > 0:
            issues.append('redundant_sentences')
            return max(0.5, 0.95 - redundancy_count * 0.15)
        
        return 0.95
    
    def _check_structure(self, query: str, response: str, intent: str, issues: List[str]) -> float:
        """Check if response structure matches intent (e.g., definition query has definition)."""
        if intent == 'definition' or re.match(r'^\s*what\s+is\s+', query, re.I):
            # Should start with a definition-like sentence
            first = response.split('.')[0] if '.' in response else response
            if not _DEF_RE.search(first):
                issues.append('missing_definition_structure')
                return 0.6
        
        if intent == 'howto':
            # Should have numbered or structured steps
            if not re.search(r'^\s*\d+[.)]\s+', response, re.MULTILINE):
                issues.append('missing_step_structure')
                return 0.6
        
        return 0.95


# ═════════════════════════════════════════════════════════════════════════════
# § REFINER — Targeted Response Improvement
# ═════════════════════════════════════════════════════════════════════════════

class ResponseRefiner:
    """
    Applies targeted fixes to improve response quality.
    Works in conjunction with ResponseDiscriminator to iteratively polish responses.
    """
    
    def refine(self, query: str, response: str, issues: List[str], subject_terms: Set[str]) -> str:
        """Apply fixes based on discriminator feedback."""
        refined = response
        
        # ── Fix 1: Orphaned pronouns ──────────────────────────────────────────
        if 'orphaned_pronoun_in_first_sentence' in issues:
            refined = self._fix_orphaned_pronouns(query, refined, subject_terms)
        
        # ── Fix 2: Add transitions for coherence ─────────────────────────────
        if 'lacks_transitions' in issues:
            refined = self._add_transitions(refined)
        
        # ── Fix 3: Remove redundant sentences ────────────────────────────────
        if 'redundant_sentences' in issues:
            refined = self._remove_redundancy(refined)
        
        # ── Fix 4: Missing definition structure ──────────────────────────────
        if 'missing_definition_structure' in issues:
            refined = self._improve_definition_structure(refined, subject_terms)
        
        return refined
    
    def _fix_orphaned_pronouns(self, query: str, response: str, subject_terms: Set[str]) -> str:
        """Replace vague pronouns with explicit subject from query."""
        sentences = re.split(r'([.!?]+\s+)', response)
        if not sentences:
            return response
        
        # Extract likely subject from query (preserve order from query, not alphabetical)
        query_lower = query.lower()
        # Get terms in the order they appear in the query
        ordered_terms = []
        for term in subject_terms:
            if term in query_lower:
                ordered_terms.append((query_lower.index(term), term))
        ordered_terms.sort()
        subject = ' '.join(t[1] for t in ordered_terms[:3])
        
        if not subject:
            return response
        
        # Fix first sentence only (most critical)
        first = sentences[0]
        # Replace leading "It is" / "They are" with explicit subject
        first = re.sub(r'^\s*It\s+is\b', f'{subject.title()} is', first, flags=re.I)
        first = re.sub(r'^\s*They\s+are\b', f'{subject.title()} are', first, flags=re.I)
        first = re.sub(r'^\s*This\s+is\b', f'{subject.title()} is', first, flags=re.I)
        
        sentences[0] = first
        return ''.join(sentences)
    
    def _add_transitions(self, response: str) -> str:
        """Add transition words between paragraphs or key sentences."""
        # Split into paragraphs
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return response
        
        # Add transitions to second paragraph if it lacks one
        second = paragraphs[1]
        has_transition = re.match(
            r'^\s*(however|moreover|furthermore|additionally|in addition|'
            r'for example|for instance|similarly|in contrast)\b',
            second, re.I
        )
        
        if not has_transition:
            # Check if it's an example or elaboration
            if _EXAMPLE_RE.search(second):
                paragraphs[1] = 'For example, ' + second[0].lower() + second[1:]
            else:
                paragraphs[1] = 'Additionally, ' + second[0].lower() + second[1:]
        
        return '\n\n'.join(paragraphs)
    
    def _remove_redundancy(self, response: str) -> str:
        """Remove highly redundant sentences."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip() and len(s) > 20]
        if len(sentences) < 2:
            return response
        
        unique = [sentences[0]]
        for s in sentences[1:]:
            words_s = set(re.findall(r'\w+', s.lower())) - _STOP
            is_redundant = False
            for u in unique:
                words_u = set(re.findall(r'\w+', u.lower())) - _STOP
                if words_s and words_u:
                    overlap = len(words_s & words_u) / min(len(words_s), len(words_u))
                    # More aggressive redundancy threshold
                    if overlap > 0.60:  # Changed from 0.75 to 0.60
                        is_redundant = True
                        break
            if not is_redundant:
                unique.append(s)
        
        # Limit to best 4 sentences to avoid overwhelming user
        if len(unique) > 4:
            unique = unique[:4]
        
        return '. '.join(unique) + ('.' if unique else '')
    
    def _improve_definition_structure(self, response: str, subject_terms: Set[str]) -> str:
        """Reorder sentences to put definition first if present but not leading."""
        sentences = [s.strip() for s in re.split(r'([.!?]+)', response) if s.strip()]
        
        # Find first definition-like sentence
        def_idx = None
        for i in range(0, len(sentences), 2):  # even indices are sentences (odd are punctuation)
            if i < len(sentences) and _DEF_RE.search(sentences[i]):
                def_idx = i
                break
        
        if def_idx is not None and def_idx > 0:
            # Move definition to front
            definition = sentences[def_idx:def_idx + 2]  # sentence + punctuation
            rest = sentences[:def_idx] + sentences[def_idx + 2:]
            return ''.join(definition + rest)
        
        return response


# ─────────────────────────────────────────────────────────────────────────────

class ResponseGenerator:
    """
    Generates responses using a RAG-style extraction pipeline with
    generator-discriminator refinement:
      1. Clean and filter all sentences from retrieved content
      2. Score each sentence for subject relevance
      3. Categorize: DEFINITION → ELABORATION → EXAMPLE
      4. Build coherent response
      5. **Evaluate quality** (discriminator)
      6. **Refine if needed** (iterative polishing until quality threshold met)
    """

    def __init__(self):
        self.discriminator = ResponseDiscriminator()
        self.refiner = ResponseRefiner()
        self._refinement_threshold = 0.78  # quality score to accept response
        self._use_llm = config.USE_LOCAL_LLM and LLM_AVAILABLE
        self._use_simple_qa = SIMPLE_QA_AVAILABLE
        self._use_minigpt = MINIGPT_AVAILABLE
        self._use_learning = LEARNING_AVAILABLE
        
        # Initialize learning system
        if self._use_learning:
            self.learning = get_learning_system()
            logger.info("Learning system enabled - μSage will improve with usage!")
        else:
            self.learning = None
        
        if self._use_simple_qa:
            logger.info("Simple Q&A (zero-dependency) enabled")
        if self._use_minigpt:
            logger.info("MiniGPT (lightweight) enabled for answer generation")
        elif self._use_llm:
            logger.info("Local LLM enabled for answer generation")
        else:
            if config.USE_LOCAL_LLM and not LLM_AVAILABLE:
                logger.warning("Local LLM requested but not available - install: pip install transformers torch")
            logger.info("Using extraction-based response generation")

    # ── Public: from web sources ─────────────────────────────────────────────

    def generate_from_sources(
        self,
        query: str,
        sources: List[Dict[str, str]],
        conversation_context: str = "",
        intent: str = 'general',
    ) -> str:
        if not sources:
            return self._no_results(query)

        chunks = []
        for src in sources[:4]:
            content = src.get("content", src.get("snippet", ""))
            if content.strip():
                chunks.append(content[:1800])
        if not chunks:
            return self._no_results(query)

        combined = "\n\n".join(chunks)

        # ── Try Learned Q&A first (personalized, from past interactions) ────
        if self._use_learning:
            logger.info(f"[GENERATE_FROM_SOURCES] Checking learned Q&A for: '{query}'")
            learned_answer = self.learning.get_learned_answer(query)
            if learned_answer:
                logger.info(f"[GENERATE_FROM_SOURCES] ✅ Using learned Q&A: {learned_answer[:50]}...")
                self.learning.log_query(query, learned_answer, "learned", True)
                return learned_answer
            else:
                logger.info("[GENERATE_FROM_SOURCES] ❌ No learned answer found")
        else:
            logger.info("[GENERATE_FROM_SOURCES] ❌ Learning is DISABLED (_use_learning=False)")

        # ── Try Simple Q&A (instant, zero dependencies) ─────────────────────
        if self._use_simple_qa:
            simple_answer = simple_qa.get_answer(query, threshold=0.75)
            if simple_answer:
                logger.info("Using Simple Q&A answer (instant)")
                if self._use_learning:
                    self.learning.log_query(query, simple_answer, "simple_qa", True)
                return simple_answer

        # ── Try MiniGPT (lightweight, zero dependencies) ────────────────────
        if self._use_minigpt:
            try:
                minigpt_answer = minigpt.generate_answer(query, max_length=80)
                if minigpt_answer and len(minigpt_answer) > 20:
                    # Validate MiniGPT answer
                    if self._is_response_valid(query, minigpt_answer):
                        logger.info("Using MiniGPT answer (lightweight model)")
                        if self._use_learning:
                            self.learning.log_query(query, minigpt_answer, "minigpt", True)
                        return minigpt_answer
                    else:
                        logger.debug("MiniGPT answer failed validation, trying next method")
            except Exception as e:
                logger.debug(f"MiniGPT generation failed: {e}")

        # ── Try Local LLM (Qwen - heavier but more capable) ─────────────────
        if self._use_llm:
            try:
                llm_answer = local_llm.generate_answer(
                    query=query,
                    context=combined,
                    max_length=config.LLM_MAX_TOKENS
                )
                if llm_answer and len(llm_answer) > 30:
                    # Validate LLM answer
                    if self._is_response_valid(query, llm_answer):
                        logger.info("Using LLM-generated answer")
                        if self._use_learning:
                            self.learning.log_query(query, llm_answer, "llm", True)
                        return llm_answer
                    else:
                        logger.warning("LLM answer failed validation, falling back to extraction")
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, falling back to extraction")

        # ── Fallback to extraction-based approach ───────────────────
        # Route to the right extractor
        if intent == 'howto':
            result = self._extract_howto(combined)
            if result:
                return result

        if intent == 'comparison':
            return self._extract_comparison(query, combined)

        # ── Generator-Discriminator Refinement Loop ──────────────────────────
        # Generate initial response
        response, subject_terms = self._rag_extract(query, combined, intent)
        
        # ── Sanity Check: Detect garbage/off-topic/wrong-language responses ──────
        if not self._is_response_valid(query, response):
            return self._honest_cant_answer(query)
        
        # Iteratively refine (max 2 iterations)
        for iteration in range(2):
            quality = self.discriminator.evaluate(query, response, intent)
            
            # Accept if quality is good enough
            if quality['overall'] >= self._refinement_threshold:
                break
            
            # Apply targeted refinements
            if quality['issues']:
                response = self.refiner.refine(query, response, quality['issues'], subject_terms)
        
        # Log query with extraction method
        if self._use_learning:
            self.learning.log_query(query, response, "extraction", len(response) > 30)
        
        return response

    # ── Public: from cache ───────────────────────────────────────────────────

    def generate_from_cache(
        self,
        query: str,
        cached_results: List[Tuple[str, float, dict]],
        conversation_context: str = "",
    ) -> Optional[str]:
        if not cached_results:
            return None
        content, similarity, _ = cached_results[0]
        if similarity < config.MIN_SIMILARITY_THRESHOLD:
            return None
        # Cached results also go through refinement
        response, subject_terms = self._rag_extract(query, content, 'general')
        
        # Sanity check
        if not self._is_response_valid(query, response):
            return None  # Fall through to web search
        
        for _ in range(1):  # Single refinement pass for cache
            quality = self.discriminator.evaluate(query, response, 'general')
            if quality['overall'] >= self._refinement_threshold or not quality['issues']:
                break
            response = self.refiner.refine(query, response, quality['issues'], subject_terms)
        return response

    # ── Feedback & Learning ──────────────────────────────────────────────────

    def record_feedback(self, query: str, answer: str, helpful: bool, comment: str = ""):
        """
        Record user feedback to improve future answers
        
        Args:
            query: The question asked
            answer: The answer provided
            helpful: Whether answer was helpful
            comment: Optional comment
        """
        if self._use_learning:
            self.learning.record_feedback(query, answer, helpful, comment)
    
    def get_stats(self) -> str:
        """Get learning statistics"""
        if self._use_learning:
            return self.learning.get_stats_summary()
        return "Learning system not available"
    
    def export_learned_knowledge(self, output_file: str = "learned_qa.py"):
        """Export learned Q&A pairs for integration"""
        if self._use_learning:
            self.learning.export_learned_qa(output_file)
        else:
            print("Learning system not available")

    # ── Core RAG pipeline ────────────────────────────────────────────────────

    def _is_response_valid(self, query: str, response: str) -> bool:
        """
        Sanity check: return False if response is garbage/off-topic/wrong language.
        Prevents returning nonsense to users.
        """
        if not response or len(response.strip()) < 20:
            return False
        
        # Check 1: Is the response in English when query is English?
        if not _is_mostly_english(response):
            logger.warning(f"Response appears to be in non-English language")
            return False
        
        # Check 2: Is response relevant to query?
        relevance = _response_relevance(query, response)
        if relevance < 0.15:  # Less than 15% query term overlap
            logger.warning(f"Response relevance too low: {relevance:.2f}")
            return False
        
        # Check 3: Code request but response has no code?
        if _CODE_REQUEST_RE.search(query):
            # Should contain code-like content (indentation, brackets, semicolons)
            has_code = bool(re.search(r'(^\s{4,}\w|```|def\s+\w+\(|class\s+\w+|\{[\s\S]{10,}\})', response, re.MULTILINE))
            if not has_code:
                logger.warning("Code request but response has no code")
                return False
        
        return True
    
    def _honest_cant_answer(self, query: str) -> str:
        """
        Return an honest 'cannot answer' message instead of garbage.
        Detects type of query to provide helpful guidance with polite tone.
        """
        if _CODE_REQUEST_RE.search(query):
            return (
                "I apologize, but I'm currently unable to generate code examples for this specific request.\n\n"
                "For programming help, I'd recommend:\n"
                "  • Official documentation for your language/framework\n"
                "  • Stack Overflow for specific coding questions\n"
                "  • GitHub for example implementations\n\n"
                "However, I'd be happy to explain programming concepts, algorithms, or help you "
                "understand how something works! Would you like me to explain the concept instead?"
            )
        
        # Use polite response from builtin knowledge
        return (
            f"I apologize, but I couldn't find a reliable answer for your question.\n\n"
            "The search results I found weren't clear or relevant enough to provide you "
            "with accurate information. I always prefer to be honest rather than give you "
            "uncertain or potentially incorrect information.\n\n"
            "Here are some suggestions:\n"
            "  • Try rephrasing your question in different words\n"
            "  • Be more specific about what aspect you're interested in\n"
            "  • Break complex questions into smaller, focused parts\n"
            "  • Check if the topic might be very recent or specialized\n\n"
            "Feel free to ask me something else, or rephrase this question — I'm here to help!"
        )

    def _rag_extract(self, query: str, text: str, intent: str) -> Tuple[str, Set[str]]:
        """
        RAG-style extraction:
          1. Extract subject keywords from query
          2. Score all sentences by subject relevance
          3. Pick best DEFINITION + ELABORATION + EXAMPLE sentences
          4. Return them in coherent reading order
        
        Returns: (response_text, subject_terms_set)
        """
        subject = self._query_subject(query)
        subject_terms = _sentence_terms(subject) if subject else _sentence_terms(query)

        sentences = _split_sentences(text)
        if not sentences:
            clean = re.sub(r'\s+', ' ', text).strip()
            return clean[:600], subject_terms

        # Score each sentence
        scored: List[Tuple[float, str, str]] = []  # (score, category, sentence)
        for sent in sentences:
            terms   = _sentence_terms(sent)
            overlap = len(subject_terms & terms)
            if overlap == 0:
                continue

            base = overlap + (overlap / max(len(subject_terms), 1))

            # Category bonuses
            if _DEF_RE.search(sent):
                cat   = 'definition'
                score = base + 3.0
            elif _EXAMPLE_RE.search(sent):
                cat   = 'example'
                score = base + 1.0
            elif _ELABORATION_RE.search(sent):
                cat   = 'elaboration'
                score = base + 1.5
            else:
                cat   = 'general'
                score = base

            # Length sweet-spot bonus (50-220 chars)
            if 50 <= len(sent) <= 220:
                score += 0.5

            scored.append((score, cat, sent))

        if not scored:
            # Fall back to top-3 sentences from text regardless of terms
            fallback = sentences[:3]
            return ' '.join(fallback), subject_terms

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build response: one definition, up to 2 elaborations, one example
        definition   : Optional[str] = None
        elaborations : List[str]     = []
        example      : Optional[str] = None
        general_pool : List[str]     = []
        seen_keys    : Set[str]      = set()

        def _add(s: str) -> bool:
            k = s[:55].lower()
            if k in seen_keys:
                return False
            seen_keys.add(k)
            return True

        for score, cat, sent in scored:
            if cat == 'definition' and definition is None:
                if _add(sent):
                    definition = sent
            elif cat == 'elaboration' and len(elaborations) < 2:  # Reduced from 3 to 2
                if _add(sent):
                    elaborations.append(sent)
            elif cat == 'example' and example is None:
                if _add(sent):
                    example = sent
            elif cat == 'general' and len(general_pool) < 1:  # Reduced from 2 to 1
                if _add(sent):
                    general_pool.append(sent)

        # Fill gaps if thin
        if definition is None and general_pool:
            definition = general_pool.pop(0)
        if len(elaborations) < 1:  # Changed from 2 to 1
            elaborations.extend(general_pool)

        # Assemble in reading order: definition → elaboration → example
        parts: List[str] = []
        if definition:
            parts.append(definition)
        parts.extend(elaborations[:2])  # Limit to 2 elaborations
        if example and len(parts) < 3:  # Reduced from 4 to 3
            parts.append(example)

        if not parts:
            return ' '.join(sentences[:3]), subject_terms  # Reduced from 4 to 3

        # For news/factual, restore the original text order so it reads naturally
        if intent in ('news', 'factual'):
            sent_pos = {s: i for i, s in enumerate(sentences)}
            parts.sort(key=lambda s: sent_pos.get(s, 9999))

        return '\n\n'.join(self._group_paragraphs(parts)), subject_terms

    def _group_paragraphs(self, sentences: List[str]) -> List[str]:
        """
        Group a flat list of sentences into 1-2 readable paragraphs.
        First sentence is the lead paragraph.
        Remaining sentences form the body paragraph.
        """
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences
        lead = sentences[0]
        body = ' '.join(sentences[1:])
        return [lead, body]

    def _query_subject(self, query: str) -> str:
        """
        Strip question / task wording to get the core subject term.
        e.g. "explain calculus"          → "calculus"
             "what is machine learning"  → "machine learning"
             "how to reverse a list in python" → "reverse list python"
        """
        q = query.strip()
        q = re.sub(
            r'^(what\s+is\s+(a\s+|an\s+|the\s+)?|'
            r'explain\s+(what\s+is\s+)?|define\s+|meaning\s+of\s+|'
            r'tell\s+me\s+about\s+|describe\s+|'
            r'how\s+(do|does|did|do\s+i|to)\s+|'
            r'who\s+(is|was|invented|created|made|founded)\s+|'
            r'when\s+(was|did|is)\s+|where\s+(is|was|are)\s+)',
            '', q, flags=re.I,
        ).strip()
        # Strip trailing punctuation
        q = q.rstrip('?.!')
        return q

    # ── Howto extractor ──────────────────────────────────────────────────────

    def _extract_howto(self, text: str) -> Optional[str]:
        """
        Extract numbered/bulleted steps.
        Returns formatted steps string, or None if none found.
        """
        step_re = re.compile(
            r'^\s*(?:(\d+)[.)]\s+|[-*•▸→]\s+|step\s+\d+[.:]\s*)',
            re.I | re.MULTILINE,
        )
        steps = []
        for line in text.splitlines():
            line = line.strip()
            if step_re.match(line):
                content = step_re.sub('', line).strip()
                content = _clean(content)
                if len(content.split()) >= 4:
                    steps.append(content.rstrip('.') + '.')
            if len(steps) >= 8:
                break

        if len(steps) < 2:
            # Try verb-imperative sentences (Install, Open, Run, Click…)
            verb_re = re.compile(
                r'^(Install|Open|Run|Click|Go\s+to|Navigate|Select|Enter|'
                r'Type|Create|Add|Edit|Save|Download|Set|Configure|'
                r'Make\s+sure|Ensure|Check)\b',
            )
            for sent in _split_sentences(text):
                if verb_re.match(sent):
                    steps.append(sent.rstrip('.') + '.')
                if len(steps) >= 8:
                    break

        if len(steps) < 2:
            return None

        return '\n'.join(f"{i + 1}. {s}" for i, s in enumerate(steps[:8]))

    # ── Comparison extractor ─────────────────────────────────────────────────

    def _extract_comparison(self, query: str, text: str) -> str:
        """
        For vs/comparison queries: prefer sentences mentioning both subjects.
        """
        m = re.search(
            r'([\w\s]+?)\s+(?:vs\.?|versus|compared?\s+to|vs\s+)\s+([\w\s]+)',
            query, re.I,
        )
        stop = {'a', 'an', 'the', 'is', 'are', 'and', 'or', 'of', 'to'}
        a_words: Set[str] = set()
        b_words: Set[str] = set()
        if m:
            a_words = set(re.findall(r'\w+', m.group(1).lower())) - stop
            b_words = set(re.findall(r'\w+', m.group(2).lower())) - stop

        both, one, other = [], [], []
        for sent in _split_sentences(text):
            terms = _sentence_terms(sent)
            has_a = bool(a_words & terms)
            has_b = bool(b_words & terms)
            if has_a and has_b:
                both.append(sent)
            elif has_a or has_b:
                one.append(sent)
            else:
                other.append(sent)

        pool = (both[:4] + one[:3])[:6]
        if len(pool) < 2:
            pool = other[:4]
        if not pool:
            return self._rag_extract(query, text, 'general')

        seen: Set[str] = set()
        unique = [s for s in pool if not seen.__contains__(s[:55].lower())
                  and not seen.add(s[:55].lower())]  # type: ignore[func-returns-value]
        return '\n\n'.join(self._group_paragraphs(unique[:5]))

    # ── Utilities ────────────────────────────────────────────────────────────

    def _no_results(self, query: str) -> str:
        return (
            f"I couldn't find a good answer for \"{query}\" right now.\n"
            "Try rephrasing, or ask me something more specific."
        )

    def generate_greeting(self) -> str:
        return "Hello! I'm μSage.\nAsk me anything to get started!"

    def generate_offline_message(self) -> str:
        return "Hello! I'm μSage.\nAsk me anything to get started!"

    def generate_error_response(self, error_type: str = "general") -> str:
        if error_type == "search":
            return (
                "The web search didn't return results right now.\n"
                "Please check your connection or try again in a moment."
            )
        return "Something went wrong. Try rephrasing your question."

    def format_stats_response(self, stats: Dict) -> str:
        lines = ["📊 μSage Statistics\n" + "─" * 40]

        if "knowledge" in stats:
            kb = stats["knowledge"]
            lines.append("\nKnowledge Base")
            lines.append(f"  Entries    : {kb.get('total_entries', 0)}")
            lines.append(f"  Usefulness : {kb.get('avg_usefulness', 0):.0%}")
            most = kb.get('most_accessed')
            if most:
                lines.append(f"  Top topic  : {most[:60]}{'...' if len(most) > 60 else ''}")

        if "embeddings" in stats:
            emb = stats["embeddings"]
            lines.append("\nSemantic Index")
            lines.append(f"  Indexed    : {emb.get('total_embeddings', 0)} items")
            lines.append(f"  Engine     : {'FAISS' if emb.get('using_faiss') else 'NumPy'}")

        if "conversation" in stats:
            conv = stats["conversation"]
            lines.append("\nConversation")
            lines.append(f"  Messages   : {conv.get('total_messages', 0)}")
            started = conv.get('session_start', '')
            if started:
                lines.append(f"  Started    : {started[:19]}")

        return "\n".join(lines)


