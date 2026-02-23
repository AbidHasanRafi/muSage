"""Main μSage (MuSage) Agent - Orchestrates all components"""

import re
import math
import logging
from datetime import datetime
from typing import Optional

from .search import WebSearcher
from .scraper import RobustWebScraper
from .knowledge import KnowledgeBase
from .embeddings import EmbeddingEngine
from .conversation import ConversationManager
from .responses import ResponseGenerator
from .builtin_knowledge import get_builtin_answer
from . import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# § 1  CONVERSATIONAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

_GREET_RE  = re.compile(
    r'^\s*(hi+|hello+|hey+|howdy|sup|hiya|yo|'
    r'good\s*(morning|afternoon|evening|night|day))\s*[!?.,]?\s*$', re.I)
_BYE_RE    = re.compile(
    r'^\s*(bye+|goodbye|see\s*you|later|cya|farewell|good\s*night|gn|'
    r'take\s*care|ttyl|peace)\s*[!?.,]?\s*$', re.I)
_THANKS_RE = re.compile(
    r'^\s*(thank(s| you)+|ty|thx|cheers|appreciate|much\s+appreciated|'
    r'many\s+thanks|ta|gracias)\s*(it|that|you|a\s+lot|so\s+much)?\s*[!.,]?\s*$', re.I)
_ACK_RE    = re.compile(
    r'^\s*(ok(ay)?|sure|got\s*it|understood|alright|cool|great|nice|wow|'
    r'k|yep|yeah|yup|go\s*ahead|proceed|sounds\s+good|perfect|noted|'
    r'right|makes\s+sense|i\s+see)\s*[!.,]?\s*$', re.I)
_ABOUT_RE  = re.compile(r'(who|what)\s*(are|is)\s*(you|musage|mu\s*sage)\??', re.I)
_POSITIVE_RE = re.compile(
    r'^\s*(yes+|yeah+|yep|yup|sure|go\s*(ahead|for\s*it)|do\s*it|'
    r'ok(ay)?|absolutely|definitely|please|of\s*course|'
    r'sounds?\s*good|why\s*not|affirmative|correct|right|'
    r'search\s*(it|that|please)?|find\s*(it|that)?|look\s*(it\s*up)?|'
    r'please\s*search|yes\s*please|go\s*on)\s*[!.,]?\s*$', re.I)

_NEGATIVE_RE = re.compile(
    r'^\s*(no+|nope|nah|don\'t|dont|no\s*thanks|not\s*(now|really)|'
    r'cancel|stop|skip|forget\s*it|never\s*mind|nevermind|'
    r'don\'t\s*(search|look)|no\s*need|i\'m\s*good)\s*[!.,]?\s*$', re.I)

_HELP_RE   = re.compile(
    r'^\s*('
    r'help|'
    r'(i\s+)?(need|want|could\s+use|require)\s+(some\s+|a\s+(bit|lot)\s+of\s+|your\s+)?help|'
    r'can\s+(you|i\s+get)\s+(some\s+)?help|'
    r'how\s+can\s+you\s+help(\s+me)?|'
    r'what\s+can\s+you\s+do(\s+for\s+me)?'
    r')\s*[?!.]?\s*$', re.I)

# Wh-question or explain prefix — always treat as a real searchable query
_QUESTION_RE = re.compile(
    r'^\s*(what|how|why|when|where|who|which|explain|tell me about|describe|define|'
    r'what\'s|what is|how do|how does|what are)\b',
    re.I,
)

# Task-request prefixes that may be vague
_TASK_PREFIX_RE = re.compile(
    r'^\s*(i\s+need\s+(you\s+to\s+|help\s+with\s+|to\s+)?|'
    r'can\s+you\s+|could\s+you\s+|please\s+|'
    r'help\s+me\s+(with\s+|do\s+|on\s+|understand\s+)?|'
    r'do\s+(some\s+|a\s+few\s+|the\s+|an?\s+)?|perform\s+(some\s+)?|'
    r'i\s+want\s+(you\s+to\s+|to\s+)?|show\s+me\s+(how\s+to\s+)?|'
    r'give\s+me\s+|i\s+need\s+help\s+(with\s+)?)\s*',
    re.I,
)

# Generic nouns that signal an under-specified subject
_GENERIC_SUBJECTS = {
    'operations', 'operation', 'tasks', 'task', 'stuff', 'things', 'thing',
    'problems', 'problem', 'exercises', 'exercise', 'calculations', 'calculation',
    'work', 'questions', 'question', 'examples', 'example', 'maths', 'math',
    'arithmetic', 'arithmetics', 'algebra', 'calculus', 'coding', 'programming',
    'science', 'biology', 'chemistry', 'physics', 'history', 'geography',
    'something', 'anything', 'it', 'that', 'topics', 'topic', 'writing', 'essay',
    'help', 'assistance', 'support', 'advice', 'guidance', 'information', 'info',
}

_CONVERSATIONAL_RESPONSES = {
    'greet':  "Hello! 👋 What can I help you with?",
    'bye':    "Goodbye! Come back anytime. 👋",
    'thanks': "You're welcome! 😊",
    'ack':    "Got it! What would you like to know?",
    'about': (
        "I'm μSage — your intelligent assistant.\n"
        "Ask me anything: facts, explanations, current topics, or how-tos."
    ),
    'help': (
        "Here's what I can help with:\n"
        "  • Facts & definitions — 'What is machine learning?'\n"
        "  • How-to guides      — 'How to reverse a list in Python'\n"
        "  • Current events     — 'Latest news about AI'\n"
        "  • Comparisons        — 'Python vs JavaScript'\n"
        "  • Math               — '25 * 4', '15% of 200', 'sqrt of 144'\n"
        "  • Unit conversions   — '100°F to Celsius', '5 km to miles'\n"
        "  • Date & time        — 'What is today\'s date?'\n"
        "\nJust ask naturally!"
    ),
}


def _classify_conversational(query: str) -> Optional[str]:
    """Return a conversational-type key, or None if it's a real query."""
    q = query.strip()
    if _GREET_RE.match(q):        return 'greet'
    if _BYE_RE.match(q):          return 'bye'
    if _THANKS_RE.match(q):       return 'thanks'
    if _ACK_RE.match(q):          return 'ack'
    if _ABOUT_RE.search(q):       return 'about'
    if _HELP_RE.match(q):         return 'help'
    if _SMALL_TALK_RE.match(q):   return 'smalltalk'
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

_HOWTO_RE = re.compile(
    r'^\s*(how\s+(do|can|to|do\s+i|does\s+one|should\s+i)\s+|'
    r'steps?\s+(to|for)\s+|ways?\s+to\s+|guide\s+(to|for)\s+|'
    r'tutorial\s+(on|for)\s+|learn\s+(how\s+to\s+)?|'
    r'how\s+(to\s+)?(make|build|create|write|code|install|fix|solve|get|do|use|set\s+up|run|deploy))\b',
    re.I)

_DEFINITION_RE = re.compile(
    r'^\s*(what\s+is\s+(a\s+|an\s+|the\s+)?|define\s+|definition\s+of\s+|'
    r'meaning\s+of\s+|explain\s+(what\s+is\s+)?|what\s+does\s+\w+\s+mean|'
    r'describe\s+(what\s+is\s+)?)\b',
    re.I)

_COMPARISON_RE = re.compile(
    r'\b(vs\.?|versus|difference\s+between|compare\s*(to|with)?|compared\s+to|'
    r'better\s+than|worse\s+than|pros\s+and\s+cons|similarities|'
    r'which\s+is\s+better|what\s+is\s+the\s+difference)\b',
    re.I)

_NEWS_RE = re.compile(
    r'\b(latest|recent|current|news\s+about|update\s+on|'
    r'what.s\s+happening|what\s+happened|breaking|just\s+released|'
    r'new\s+in|announced|trending)\b',
    re.I)

_FOLLOWUP_RE = re.compile(
    r'^\s*(tell\s+me\s+more|elaborate|more\s+(about|on|info|details?)|'
    r'what\s+about\s+(its?|the|their|that|this)\s+\w+|go\s+on|continue|'
    r'can\s+you\s+explain\s+(that|more|further)|explain\s+(that|more|further)|'
    r'what\s+else|what\s+does\s+that\s+mean|expand\s+on\s+that|'
    r'give\s+me\s+more|i\s+want\s+to\s+know\s+more|and\s+what\s+about)\b',
    re.I)

_RECOMMENDATION_RE = re.compile(
    r'\b(best\b|recommend|top\s+\d+|should\s+i\s+(use|choose|get|buy|try|learn)|'
    r'which\s+(is|are)\s+better|most\s+(popular|used|common|efficient))\b',
    re.I)

_FACTUAL_RE = re.compile(
    r'^\s*(who\s+(is|was|invented|discovered|created|wrote|founded|made)|'
    r'when\s+(was|did|is|were)|where\s+(is|was|are|were)|'
    r'what\s+(is\s+the\s+(capital|population|height|distance|speed|weight|age|formula|symbol)|'
    r'are\s+the\s+(main|key|top|best|types|kinds)|was\s+the\s+(first|last|biggest))|'
    r'how\s+(many|much|long|far|tall|big|old|fast)\b)',
    re.I)

_OPINION_RE = re.compile(
    r'^\s*(do\s+you\s+(think|believe|feel|like|prefer)|'
    r'what\s+do\s+you\s+think\s+about|in\s+your\s+opinion|'
    r'your\s+(thoughts?|view|take)\s+on)\b',
    re.I)


def _classify_intent(query: str) -> str:
    """
    Classify into: followup | comparison | news | howto | definition |
                   factual | recommendation | opinion | general
    """
    q = query.strip()
    if _FOLLOWUP_RE.match(q):         return 'followup'
    if _COMPARISON_RE.search(q):      return 'comparison'
    if _NEWS_RE.search(q):            return 'news'
    if _HOWTO_RE.match(q):            return 'howto'
    if _DEFINITION_RE.match(q):       return 'definition'
    if _FACTUAL_RE.match(q):          return 'factual'
    if _RECOMMENDATION_RE.search(q):  return 'recommendation'
    if _OPINION_RE.match(q):          return 'opinion'
    return 'general'


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  VAGUE-TASK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Short openers that should never reach the web
_SMALL_TALK_RE = re.compile(
    r'^\s*(('
    r'what(\'s|\s+is)\s+(up|new|going\s+on)|'
    r'how\s+(are\s+you|is\s+it\s+going|do\s+you\s+do)|'
    r'are\s+you\s+(ok|okay|there|ready|working)|'
    r'(good|bad|nice|cool|awesome|amazing|wow|interesting|'  
    r'i\s+(see|know|get\s+it|understand|got\s+it))|'
    r'(yes+|no+|nope|nah|absolutely|definitely|of\s+course|sure\s+thing)|'
    r'never\s+mind|forget\s+it|never\s+mind|stop|clear|reset'
    r'))\s*[!?.,]?\s*$', re.I)

_CONVERSATIONAL_RESPONSES['smalltalk'] = "I'm here and ready to help! What would you like to know?"

_CLARIFY_MAP = {
    'help':         "What would you like help with? I can answer questions, explain topics, do math, help with how-tos, and more.",
    'assistance':   "What would you like help with? I can answer questions, explain topics, do math, help with how-tos, and more.",
    'arithmetic':   "What would you like to work through — addition, subtraction, multiplication, division, or a specific problem?",
    'arithmetics':  "What would you like to work through — addition, subtraction, multiplication, division, or a specific problem?",
    'math':         "What kind of math? Give me a specific problem or topic (e.g. fractions, algebra, geometry).",
    'maths':        "What kind of math? Give me a specific problem or topic (e.g. fractions, algebra, geometry).",
    'algebra':      "What algebra problem or concept? (e.g. solve for x, quadratic formula)",
    'calculus':     "What calculus topic? (e.g. derivatives, integrals, limits)",
    'coding':       "What would you like to build, or what specific problem are you trying to solve?",
    'programming':  "Which language or problem? (e.g. Python sorting, JavaScript async, SQL queries)",
    'science':      "Which area of science? (e.g. physics, biology, chemistry, astronomy)",
    'biology':      "What biology topic? (e.g. cell structure, genetics, evolution, photosynthesis)",
    'chemistry':    "What chemistry topic? (e.g. periodic table, reactions, acids and bases)",
    'physics':      "What physics topic? (e.g. Newton's laws, electricity, quantum mechanics)",
    'history':      "Which period or event in history are you curious about?",
    'geography':    "Which region or concept? (e.g. capitals, climate zones, rivers, countries)",
    'operations':   "What kind of operations? Give me a specific problem and I'll help.",
    'calculations': "What would you like to calculate? Share the numbers or the formula.",
    'writing':      "What kind of writing? (e.g. essay structure, cover letter, email, story)",
    'essay':        "What topic is the essay on? I can help with structure, arguments, or content.",
}


def _is_vague_task(query: str) -> bool:
    """
    True when the query is a task-request without enough specifics to act on.
    e.g. "I need you to do some arithmetic" → True
         "What is arithmetic?"              → False  (wh-word)
         "Explain long division"            → False  (explicit topic)
    """
    q = query.strip()
    if '?' in q:                return False
    if _QUESTION_RE.match(q):   return False
    m = _TASK_PREFIX_RE.match(q)
    if not m:                   return False
    subject = q[m.end():].strip().rstrip('.,!')
    words   = [w.lower() for w in subject.split() if w.isalpha()]
    if not words:               return True
    if len(words) <= 2:
        return any(w in _GENERIC_SUBJECTS for w in words) or len(words) == 1
    return words[-1] in _GENERIC_SUBJECTS


def _clarifying_question(query: str) -> str:
    """Build a natural, topic-aware clarifying question."""
    q = query.strip()
    m = _TASK_PREFIX_RE.match(q)
    subject = q[m.end():].strip().rstrip('.,!') if m else q
    sl = subject.lower()
    for kw, text in _CLARIFY_MAP.items():
        if kw in sl:
            return text
    if subject and sl not in _GENERIC_SUBJECTS:
        return f"Could you be more specific about what you'd like help with regarding {subject}?"
    return "Could you be more specific? What exactly would you like help with?"


# ═══════════════════════════════════════════════════════════════════════════════
# § 3b  CONTEXT THREADING — pronoun/reference resolution across turns
# ═══════════════════════════════════════════════════════════════════════════════

# Detect pronouns that refer back to a previous subject
_PRONOUN_REF_RE = re.compile(
    r'\b(it|its|they|their|them|those|these)\b',
    re.I,
)

# Bare continuation phrases — no explicit topic, just asking for more
_BARE_CONTINUATION_RE = re.compile(
    r'^\s*(tell\s+me\s+more|more\s+(about\s+it|on\s+that|info|details?)|'
    r'go\s+on|continue|elaborate|expand\s+(on\s+that)?|what\s+else|'
    r'and\s+(what|how|why|when|who)\??|keep\s+going|more\s+please|'
    r'can\s+you\s+elaborate|explain\s+more|give\s+me\s+more|'
    r'what\s+(else|more)\s+(can|do|should)\s+i|'
    r'how\s+does\s+(it|that)\s+work|what\s+is\s+(it|that)\s+(used\s+for)?|'
    r'why\s+is\s+(it|that)\s+important|what\s+are\s+(its|their)\s+uses?)'
    r'\s*[?.!]?\s*$',
    re.I,
)


def _extract_subject(query: str) -> str:
    """
    Extract the core subject noun phrase from a query for context threading.
    Strips common interrogative / task prefixes.
    e.g. "what is machine learning?"         → "machine learning"
         "explain neural networks"           → "neural networks"
         "how does Python work?"             → "Python"
         "what are the best Python frameworks" → "best Python frameworks"
    """
    q = re.sub(
        r'^\s*(what\s+is\s+(a\s+|an\s+|the\s+)?|'
        r'explain\s+(what\s+is\s+)?|define\s+|meaning\s+of\s+|'
        r'tell\s+me\s+about\s+|describe\s+|'
        r'how\s+(do|does|did|can|do\s+i|to)\s+|'
        r'who\s+(is|was|invented|created|made|founded)\s+|'
        r'when\s+(was|did|is)\s+|where\s+(is|was|are)\s+|'
        r'what\s+are\s+(the\s+)?(main\s+|key\s+|best\s+|top\s+)?|'
        r'what\s+(makes?|causes?|happens?)\s+)',
        '', query.strip(), flags=re.I,
    ).strip().rstrip('?.!')
    # Drop trailing verb tails like "work", "do", "mean"
    q = re.sub(
        r'\s+(work|do|are|mean|help|come\s+from|used\s+for)\s*\??$',
        '', q, flags=re.I,
    ).strip()
    return q or query.strip().rstrip('?.!')


# Words to strip when extracting the real subject from a vague phrase
_TOPIC_NOISE = frozenset({
    'some', 'a', 'an', 'the', 'help', 'assistance', 'with', 'in', 'on',
    'about', 'for', 'bit', 'lot', 'please', 'need', 'want', 'do', 'me',
    'you', 'i', 'my', 'your', 'their', 'little', 'more', 'any',
})


def _extract_topic_noun(phrase: str) -> str:
    """
    Strip noise words from a vague phrase and return the real topic noun.
    e.g. "some help in math" → "math"
         "programming problems" → "programming"
         "calculus" → "calculus"
    """
    words = [w for w in phrase.lower().split() if w.isalpha() and w not in _TOPIC_NOISE]
    return ' '.join(words).strip() or phrase.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  LOCAL ANSWER ENGINE  (no web required)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Date / Time / Year ────────────────────────────────────────────────────────
_DATE_RE = re.compile(
    r'(what.*(date|day).*today|today.*(date|day)|what day is (it|today)|'
    r'current date|date today|what is today|what\'s today|today\'s date)',
    re.I)
_TIME_RE = re.compile(
    r'(what.*(time|clock).*now|current time|what time is it|'
    r'time (right )?now|what\'s the time)',
    re.I)
_YEAR_RE = re.compile(
    r'(what.*(year).*now|current year|what year is it|what\'s the year)',
    re.I)

# ── Math ──────────────────────────────────────────────────────────────────────
_MATH_RE = re.compile(
    r'^\s*(what\s+is\s+|calculate\s+|compute\s+|eval\s+|solve\s+)?'
    r'([\d.]+\s*(\*\*|[+\-*/\^%])\s*[\d.]+(?:\s*(\*\*|[+\-*/\^%])\s*[\d.]+)*)\s*[=?]?\s*$',
    re.I)
_PERCENT_RE = re.compile(
    r'(what\s+is\s+)?(\d+(?:\.\d+)?)\s*(%|percent)\s+of\s+(\d+(?:\.\d+)?)',
    re.I)
_SQRT_RE  = re.compile(
    r'(what\s+is\s+)?(square\s+root|sqrt)\s+(of\s+)?(\d+(?:\.\d+)?)',
    re.I)
_POWER_RE = re.compile(
    r'(what\s+is\s+)?(\d+(?:\.\d+)?)\s+(to\s+the\s+power\s+of|raised\s+to)\s+(\d+(?:\.\d+)?)',
    re.I)

# ── Temperature ───────────────────────────────────────────────────────────────
_F_TO_C  = re.compile(r'([\-\d.]+)\s*°?\s*(f|fahrenheit)\s+(to|in|into)\s+(c|celsius|centigrade)', re.I)
_C_TO_F  = re.compile(r'([\-\d.]+)\s*°?\s*(c|celsius)\s+(to|in|into)\s+(f|fahrenheit)', re.I)
_C_TO_K  = re.compile(r'([\-\d.]+)\s*°?\s*(c|celsius)\s+(to|in|into)\s+(k|kelvin)', re.I)
_K_TO_C  = re.compile(r'([\-\d.]+)\s*°?\s*(k|kelvin)\s+(to|in|into)\s+(c|celsius)', re.I)
_K_TO_F  = re.compile(r'([\-\d.]+)\s*°?\s*(k|kelvin)\s+(to|in|into)\s+(f|fahrenheit)', re.I)

# ── Length ────────────────────────────────────────────────────────────────────
_KM_MI   = re.compile(r'([\d.]+)\s*(km|kilometers?|kilometres?)\s+(to|in|into)\s+(mi|miles?)', re.I)
_MI_KM   = re.compile(r'([\d.]+)\s*(mi|miles?)\s+(to|in|into)\s+(km|kilometers?|kilometres?)', re.I)
_CM_IN   = re.compile(r'([\d.]+)\s*(cm|centim[ei]ters?)\s+(to|in|into)\s+(in|inch|inches)', re.I)
_IN_CM   = re.compile(r'([\d.]+)\s*(in|inch|inches)\s+(to|in|into)\s+(cm|centim[ei]ters?)', re.I)
_M_FT    = re.compile(r'([\d.]+)\s*(m|meters?|metres?)\s+(to|in|into)\s+(ft|feet|foot)', re.I)
_FT_M    = re.compile(r'([\d.]+)\s*(ft|feet|foot)\s+(to|in|into)\s+(m|meters?|metres?)', re.I)

# ── Weight ────────────────────────────────────────────────────────────────────
_KG_LB   = re.compile(r'([\d.]+)\s*(kg|kilograms?)\s+(to|in|into)\s+(lb|lbs|pounds?)', re.I)
_LB_KG   = re.compile(r'([\d.]+)\s*(lb|lbs|pounds?)\s+(to|in|into)\s+(kg|kilograms?)', re.I)
_G_OZ    = re.compile(r'([\d.]+)\s*(g|grams?)\s+(to|in|into)\s+(oz|ounces?)', re.I)
_OZ_G    = re.compile(r'([\d.]+)\s*(oz|ounces?)\s+(to|in|into)\s+(g|grams?)', re.I)

# ── Speed ─────────────────────────────────────────────────────────────────────
_MPH_KPH = re.compile(r'([\d.]+)\s*(mph|miles\s+per\s+hour)\s+(to|in|into)\s+(kph|kmh|km/h|km\s+per\s+hour)', re.I)
_KPH_MPH = re.compile(r'([\d.]+)\s*(kph|kmh|km/h|kilometers?\s+per\s+hour)\s+(to|in|into)\s+(mph|miles\s+per\s+hour)', re.I)

# ── Age calculation ───────────────────────────────────────────────────────────
_AGE_RE    = re.compile(
    r'(how\s+old\s+(is|would)\s+(someone|a\s+person)|age\s+of\s+someone)'
    r'\s+(born\s+in\s+|if\s+born\s+in\s+)?(\d{4})', re.I)
_MY_AGE_RE = re.compile(r'i\s+(was|am)\s+born\s+in\s+(\d{4})', re.I)


def _fmt(val: float) -> str:
    """Format number: drop unnecessary trailing decimals."""
    if isinstance(val, float) and val == int(val) and abs(val) < 1e15:
        return str(int(val))
    return f"{val:.8g}"


def _try_local_answer(query: str) -> Optional[str]:
    """
    Return an instant local answer without touching the web.
    Covers: date/time/year, math, sqrt, power, percentages,
            temperature/length/weight/speed conversions, age.
    Returns None if the query needs the normal pipeline.
    """
    q = query.strip()

    # ── Date / Time / Year ────────────────────────────────────────────────────
    if _DATE_RE.search(q):
        n = datetime.now()
        return f"Today is {n.strftime('%A')}, {n.strftime('%B')} {n.day}, {n.year}."
    if _TIME_RE.search(q):
        return f"The current time is {datetime.now().strftime('%I:%M %p')}."
    if _YEAR_RE.search(q):
        return f"The current year is {datetime.now().year}."

    # ── Square root ───────────────────────────────────────────────────────────
    m = _SQRT_RE.search(q)
    if m:
        n = float(m.group(4))
        return (f"√{_fmt(n)} = {_fmt(math.sqrt(n))}"
                if n >= 0 else "Square root of a negative number is not real.")

    # ── Power (natural language) ──────────────────────────────────────────────
    m = _POWER_RE.search(q)
    if m:
        base, exp = float(m.group(2)), float(m.group(4))
        return f"{_fmt(base)} ^ {_fmt(exp)} = {_fmt(base ** exp)}"

    # ── Percentage ────────────────────────────────────────────────────────────
    m = _PERCENT_RE.search(q)
    if m:
        pct, base = float(m.group(2)), float(m.group(4))
        return f"{_fmt(pct)}% of {_fmt(base)} = {_fmt(pct / 100 * base)}"

    # ── Arithmetic expression ─────────────────────────────────────────────────
    m = _MATH_RE.match(q)
    if m:
        expr = m.group(2).replace('^', '**')
        try:
            allowed = set('0123456789.+-*/%() ')
            if all(c in allowed for c in expr.replace('**', '')):
                result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
                return f"{expr.strip()} = {_fmt(result)}"
        except Exception:
            pass

    # ── Temperature ───────────────────────────────────────────────────────────
    for pat, fn in [
        (_F_TO_C, lambda v: f"{_fmt(v)}°F  =  {_fmt((v - 32) * 5 / 9)}°C"),
        (_C_TO_F, lambda v: f"{_fmt(v)}°C  =  {_fmt(v * 9 / 5 + 32)}°F"),
        (_C_TO_K, lambda v: f"{_fmt(v)}°C  =  {_fmt(v + 273.15)} K"),
        (_K_TO_C, lambda v: f"{_fmt(v)} K  =  {_fmt(v - 273.15)}°C"),
        (_K_TO_F, lambda v: f"{_fmt(v)} K  =  {_fmt((v - 273.15) * 9 / 5 + 32)}°F"),
    ]:
        m = pat.search(q)
        if m:
            return fn(float(m.group(1)))

    # ── Length ────────────────────────────────────────────────────────────────
    for pat, fn in [
        (_KM_MI, lambda v: f"{_fmt(v)} km  =  {_fmt(v * 0.621371)} miles"),
        (_MI_KM, lambda v: f"{_fmt(v)} miles  =  {_fmt(v * 1.60934)} km"),
        (_CM_IN, lambda v: f"{_fmt(v)} cm  =  {_fmt(v * 0.393701)} inches"),
        (_IN_CM, lambda v: f"{_fmt(v)} inches  =  {_fmt(v * 2.54)} cm"),
        (_M_FT,  lambda v: f"{_fmt(v)} m  =  {_fmt(v * 3.28084)} ft"),
        (_FT_M,  lambda v: f"{_fmt(v)} ft  =  {_fmt(v * 0.3048)} m"),
    ]:
        m = pat.search(q)
        if m:
            return fn(float(m.group(1)))

    # ── Weight ────────────────────────────────────────────────────────────────
    for pat, fn in [
        (_KG_LB, lambda v: f"{_fmt(v)} kg  =  {_fmt(v * 2.20462)} lbs"),
        (_LB_KG, lambda v: f"{_fmt(v)} lbs  =  {_fmt(v * 0.453592)} kg"),
        (_G_OZ,  lambda v: f"{_fmt(v)} g  =  {_fmt(v * 0.035274)} oz"),
        (_OZ_G,  lambda v: f"{_fmt(v)} oz  =  {_fmt(v * 28.3495)} g"),
    ]:
        m = pat.search(q)
        if m:
            return fn(float(m.group(1)))

    # ── Speed ─────────────────────────────────────────────────────────────────
    for pat, fn in [
        (_MPH_KPH, lambda v: f"{_fmt(v)} mph  =  {_fmt(v * 1.60934)} km/h"),
        (_KPH_MPH, lambda v: f"{_fmt(v)} km/h  =  {_fmt(v * 0.621371)} mph"),
    ]:
        m = pat.search(q)
        if m:
            return fn(float(m.group(1)))

    # ── Age calculation ───────────────────────────────────────────────────────
    m = _AGE_RE.search(q)
    if m:
        year = int(m.group(5))
        age  = datetime.now().year - year
        return f"Someone born in {year} is {age} years old in {datetime.now().year}."
    m = _MY_AGE_RE.search(q)
    if m:
        year = int(m.group(2))
        age  = datetime.now().year - year
        return f"If you were born in {year}, you are {age} years old."

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  QUERY REFINER
# ═══════════════════════════════════════════════════════════════════════════════

def _refine_query(query: str, intent: str, context: str) -> str:
    """
    Transform raw user input into a cleaner search query based on intent.
    """
    q = query.strip()

    if intent == 'followup':
        lines  = [l for l in context.splitlines() if l.startswith('Assistant:')]
        if lines:
            anchor = ' '.join(lines[-1].replace('Assistant:', '').strip().split()[:10])
            return f"{anchor} {q}"
        return q

    # Strip vague task prefix
    mm = _TASK_PREFIX_RE.match(q)
    if mm:
        q = q[mm.end():].strip() or q

    if intent == 'howto' and not re.match(r'how\s+to\b', q, re.I):
        q = f"how to {q} step by step"
    elif intent == 'definition':
        q = re.sub(
            r'^(what\s+is\s+(a\s+|an\s+|the\s+)?|define\s+|meaning\s+of\s+|explain\s+)',
            '', q, flags=re.I).strip()
        q = f"what is {q}"
    elif intent == 'news':
        if not re.search(r'\b20[0-9]{2}\b', q):
            q = f"{q} {datetime.now().year}"
    elif intent == 'comparison':
        if not re.search(r'differences?|comparison', q, re.I):
            q = f"{q} comparison differences"

    return q.strip() or query.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  AGENT
# ═══════════════════════════════════════════════════════════════════════════════


class MuSageAgent:
    """
    Main agent — orchestrates search, scraping, knowledge, embeddings,
    conversation memory, and response generation.

    Decision chain in ask():
      1. Stats command
      2. Pending-clarification state
      3. Conversational shortcut (greet/bye/thanks/ack/about/help)
      4. Local instant answer (date/time/math/conversions)
      5. Vague-task detection → clarifying question
      6. Cache check (semantic similarity)
      7. Web search with intent-aware refined query
    """

    def __init__(self):
        logger.info("Initializing μSage Agent...")

        self.searcher           = WebSearcher()
        self.scraper            = RobustWebScraper()
        self.knowledge_base     = KnowledgeBase()
        self.conversation       = ConversationManager()
        self.response_generator = ResponseGenerator()
        self.embedding_engine   = EmbeddingEngine()

        self._sync_knowledge_to_embeddings()
        self.is_online  = self.searcher.is_online()
        self.last_source = ''

        # Pending-clarification state
        self._pending_topic: Optional[str] = None   # e.g. "arithmetic operations"
        self._pending_count: int = 0                # how many times we've asked

        # Confirm-before-search state
        self._pending_search: Optional[tuple] = None  # (query, intent)

        # Context threading — remembers the last discussed topic
        self._last_topic: str = ""

        logger.info("μSage Agent initialized successfully")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _sync_knowledge_to_embeddings(self):
        entries = self.knowledge_base.get_all_entries()
        if not entries:
            return
        if self.embedding_engine.get_stats()["total_embeddings"] >= len(entries):
            return
        texts     = [e.content for e in entries]
        metadatas = [{"query": e.query, "source": e.source} for e in entries]
        self.embedding_engine.add_batch_to_index(texts, metadatas)

    def _store_and_index(self, query: str, content: str, url: str, title: str = ""):
        if not content.strip():
            return
        self.knowledge_base.add_entry(query=query, content=content, source=url,
                                      metadata={"title": title})
        self.embedding_engine.add_to_index(
            text=content,
            metadata={"query": query, "source": url, "title": title},
        )

    def _do_web_search(self, query: str, context: str, intent: str = 'general') -> str:
        """Run web search + scrape for a confirmed, specific query."""
        if not self.is_online:
            return (
                "I'm not able to reach the internet right now.\n"
                "Please check your connection and try again."
            )

        refined = _refine_query(query, intent, context)
        search_results = self.searcher.search(refined)
        if not search_results:
            return self.response_generator.generate_error_response("search")

        urls    = [r["url"] for r in search_results[:3]]
        scraped = self.scraper.scrape_multiple(urls)
        sources = scraped if scraped else search_results

        for item in sources:
            self._store_and_index(
                query   = query,
                content = item.get("content", item.get("snippet", "")),
                url     = item.get("url", ""),
                title   = item.get("title", ""),
            )

        self.last_source = 'web'
        subj = _extract_subject(query)
        if subj:
            self._last_topic = subj
        return self.response_generator.generate_from_sources(query, sources, context, intent)

    # ── Public API ──────────────────────────────────────────────────────────

    def _resolve_context(self, query: str) -> str:
        """
        Rewrite the query to be self-contained by substituting pronoun
        references (it/its/they/their/them/those/these) with the last
        discussed topic, and expanding bare continuation phrases.

        Examples (last_topic = "Python"):
          "what are its frameworks?"  → "what are Python's frameworks?"
          "how does it work?"         → "how does Python work?"
          "tell me more"              → "tell me more about Python"
        """
        if not self._last_topic:
            return query

        q = query.strip()
        topic = self._last_topic

        # Bare continuation → make explicit
        if _BARE_CONTINUATION_RE.match(q):
            return f"tell me more about {topic}"

        # Only rewrite if pronouns are actually present
        if not _PRONOUN_REF_RE.search(q):
            return query

        resolved = q
        # Possessive pronouns
        resolved = re.sub(r'\bits\b', f"{topic}'s", resolved, flags=re.I)
        resolved = re.sub(r'\btheir\b', f"{topic}'s", resolved, flags=re.I)
        # Object / subject pronouns
        resolved = re.sub(r'\b(they|them)\b', topic, resolved, flags=re.I)
        resolved = re.sub(r'\bit\b', topic, resolved, flags=re.I)
        # Demonstratives only when used as subject placeholders
        resolved = re.sub(r'\b(those|these)\b', topic, resolved, flags=re.I)
        return resolved.strip()

    def _confirm_search(self, query: str, intent: str) -> str:
        """Store pending search and ask the user for confirmation."""
        self._pending_search = (query, intent)
        preview = query[:80] + ('...' if len(query) > 80 else '')
        return f"Would you like me to search for: \"{preview}\"?"

    def ask(self, query: str) -> str:
        """
        Conversation decision chain:
          1. Stats command
          2. Confirm-before-search gate  (pending_search)
          3. Pending-clarification state (pending_topic)
          4. Conversational shortcuts
          5. Local instant answer
          6. Vague-task detection
          7. Intent classify + cache check
          8. Ask for search confirmation  (sets pending_search)
        """
        self.last_source = ''
        self.conversation.add_message("user", query)
        context = self.conversation.get_context_string(n=6)

        # ── 0. Context resolution — resolve pronoun references to last topic ─
        query = self._resolve_context(query)

        # ── 1. Stats command ───────────────────────────────────────────────
        if query.strip().lower() in ("stats", "statistics", "info"):
            self._pending_topic  = None
            self._pending_search = None
            response = self._handle_stats_command()
            self.conversation.add_message("assistant", response)
            return response

        # ── 2. Confirm-before-search gate ──────────────────────────────────
        if self._pending_search is not None:
            saved_query, saved_intent = self._pending_search

            if _POSITIVE_RE.match(query.strip()):
                # User confirmed — do the search
                self._pending_search = None
                response = self._do_web_search(saved_query, context, saved_intent)
                self.conversation.add_message("assistant", response)
                return response

            if _NEGATIVE_RE.match(query.strip()):
                # User declined
                self._pending_search = None
                response = "No problem! Feel free to rephrase or ask me something else."
                self.last_source = 'conversational'
                self.conversation.add_message("assistant", response)
                return response

            # User sent something completely new — clear pending search
            # and fall through to process the new query normally
            self._pending_search = None

        # ── 3. Pending-clarification state ─────────────────────────────────
        if self._pending_topic is not None:
            conv_type = _classify_conversational(query)

            # User just said bye/thanks — clear state and respond
            if conv_type in ('bye', 'thanks'):
                self._pending_topic = None
                self._pending_count = 0
                response = _CONVERSATIONAL_RESPONSES[conv_type]
                self.last_source = 'conversational'
                self.conversation.add_message("assistant", response)
                return response

            # User said ack/yes/ok without giving content → nudge once more
            if conv_type in ('ack', 'greet') or _ACK_RE.match(query.strip()):
                if self._pending_count < 2:
                    self._pending_count += 1
                    response = f"What specifically about {self._pending_topic} would you like to know?"
                    self.conversation.add_message("assistant", response)
                    return response
                else:
                    # Give up waiting, clear state
                    self._pending_topic = None
                    self._pending_count = 0
                    response = "No problem! Feel free to ask me anything whenever you're ready."
                    self.conversation.add_message("assistant", response)
                    return response

            # User gave actual content — use as the search query, clear state
            topic = self._pending_topic
            self._pending_topic = None
            self._pending_count = 0

            # Local answer first (math / date / time) — before any search
            local = _try_local_answer(query)
            if local:
                self.last_source = 'local'
                self.conversation.add_message("assistant", local)
                return local

            # Build effective query:
            # - Explicit query (wh-question / explain / >3 words / has ?) → use as-is
            #   e.g. "explain calculus" → search "explain calculus" NOT "math: explain calculus"
            # - Short single-word reply → combine with topic for context
            #   e.g. topic="math", reply="derivatives" → "derivatives math"
            q_words = query.strip().split()
            is_explicit = (
                _QUESTION_RE.match(query.strip())
                or len(q_words) > 3
                or '?' in query
            )
            if is_explicit:
                effective_query = query.strip()
            else:
                user_noun = _extract_topic_noun(query.strip())
                # Only prefix topic if it adds distinct context
                if (user_noun and topic.lower() not in user_noun.lower()
                        and user_noun.lower() not in topic.lower()):
                    effective_query = f"{user_noun} {topic}".strip()
                else:
                    effective_query = user_noun or query.strip()

            # Check cache first
            cached = self.embedding_engine.search(effective_query, k=3)
            if cached:
                resp = self.response_generator.generate_from_cache(effective_query, cached, context)
                if resp:
                    self.last_source = 'memory'
                    subj = _extract_subject(effective_query)
                    if subj:
                        self._last_topic = subj
                    self.conversation.add_message("assistant", resp)
                    return resp

            # Ask confirmation before going to web
            response = self._confirm_search(effective_query, _classify_intent(effective_query))
            self.last_source = 'conversational'
            self.conversation.add_message("assistant", response)
            return response

        # ── 4. Conversational shortcuts ────────────────────────────────────
        conv_type = _classify_conversational(query)
        if conv_type:
            response = _CONVERSATIONAL_RESPONSES[conv_type]
            self.last_source = 'conversational'
            self.conversation.add_message("assistant", response)
            return response

        # ── 5. Local instant answers (date / time / math / conversions) ────
        local = _try_local_answer(query)
        if local:
            self.last_source = 'local'
            self.conversation.add_message("assistant", local)
            return local

        # ── 5.5 Built-in knowledge base (FAQ + word math) ─────────────────
        builtin = get_builtin_answer(query)
        if builtin:
            self.last_source = 'builtin'
            self.conversation.add_message("assistant", builtin)
            subj = _extract_subject(query)
            if subj:
                self._last_topic = subj
            return builtin

        # ── 5.6 Learned Q&A (from your past interactions) ──────────────────
        try:
            from .learning import get_learning_system
            learning = get_learning_system()
            learned_answer = learning.get_learned_answer(query)
            if learned_answer:
                logger.info("Using learned Q&A (instant retrieval from past interactions)")
                learning.log_query(query, learned_answer, "learned", True)
                self.last_source = 'learned'
                self.conversation.add_message("assistant", learned_answer)
                subj = _extract_subject(query)
                if subj:
                    self._last_topic = subj
                return learned_answer
        except ImportError:
            pass  # Learning system not available

        # ── 5.7 Simple Q&A (built-in database, zero dependencies) ──────────
        try:
            from . import simple_qa
            simple_answer = simple_qa.get_answer(query, threshold=0.75)
            if simple_answer:
                logger.info("Using Simple Q&A (instant, built-in answer)")
                try:
                    from .learning import get_learning_system
                    get_learning_system().log_query(query, simple_answer, "simple_qa", True)
                except ImportError:
                    pass
                self.last_source = 'simple_qa'
                self.conversation.add_message("assistant", simple_answer)
                subj = _extract_subject(query)
                if subj:
                    self._last_topic = subj
                return simple_answer
        except ImportError:
            pass  # Simple Q&A not available

        # ── 6. Vague-task detection → ask for clarification ────────────────
        if _is_vague_task(query):
            m = _TASK_PREFIX_RE.match(query.strip())
            raw_subject = (
                query.strip()[m.end():].strip().rstrip('.,!') if m else query.strip()
            ) or "that"
            # Store only the real noun (e.g. "math", not "some help in math")
            self._pending_topic = _extract_topic_noun(raw_subject) or raw_subject
            self._pending_count = 1
            response = _clarifying_question(query)
            self.last_source = 'conversational'
            self.conversation.add_message("assistant", response)
            return response

        # ── 7. Intent + cache check ────────────────────────────────────────
        intent = _classify_intent(query)

        cached = self.embedding_engine.search(query, k=3)
        if cached:
            response = self.response_generator.generate_from_cache(query, cached, context)
            if response:
                self.last_source = 'memory'
                subj = _extract_subject(query)
                if subj:
                    self._last_topic = subj
                self.conversation.add_message("assistant", response)
                return response

        # Update last topic for threading even when going to search
        subj = _extract_subject(query)
        if subj:
            self._last_topic = subj

        # ── 8. Ask for search confirmation ────────────────────────────────
        response = self._confirm_search(query, intent)
        self.last_source = 'conversational'
        self.conversation.add_message("assistant", response)
        return response

    # ── Special command handlers ────────────────────────────────────────────

    def _handle_stats_command(self) -> str:
        stats = {
            "knowledge":    self.knowledge_base.get_statistics(),
            "embeddings":   self.embedding_engine.get_stats(),
            "conversation": self.conversation.get_summary(),
        }
        return self.response_generator.format_stats_response(stats)

    def clear_memory(self):
        self.knowledge_base.clear()
        self.embedding_engine.clear_cache()
        self.conversation.clear()
        self._pending_topic  = None
        self._pending_count  = 0
        self._pending_search = None
        self._last_topic     = ""

    def get_greeting(self) -> str:
        if not self.is_online:
            return self.response_generator.generate_offline_message()
        return self.response_generator.generate_greeting()


