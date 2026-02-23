# μSage Complete Documentation

**Version:** 0.3.0  
**Last Updated:** February 23, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Learning System](#learning-system)
7. [Negative Feedback & Self-Correction](#negative-feedback)
8. [Answer Quality & Pipeline](#answer-quality)
9. [CLI Commands](#cli-commands)
10. [API Reference](#api-reference)
11. [Configuration](#configuration)
12. [Development Guide](#development)
13. [Troubleshooting](#troubleshooting)

---

## Overview

μSage (MuSage) is an intelligent, self-improving conversational assistant that combines:
- **Web scraping** for up-to-date information
- **Continuous learning** from user feedback
- **Local knowledge base** for instant answers
- **Zero-dependency fallback** for common questions

### Key Highlights

- **Learns from every interaction** - Builds personalized knowledge base  
- **Self-corrects from mistakes** - Removes bad answers, learns corrections  
- **Works offline** - Uses cached knowledge when internet unavailable  
- **Lightweight** - CPU-only, no GPU required  
- **Adaptive** - Gets smarter and faster over time  

---

## Features

### 1. Answer Pipeline (Priority Order)

```
User Query
    ↓
1. Learned Q&A (personalized, from your feedback)
   • Instant retrieval (<0.001s)
   • Built from answers YOU marked helpful
   • Confidence scoring (0.8-1.0)
    ↓ (no match)
2. Simple Q&A (60+ built-in topics)
   • Zero dependencies
   • Clean, curated answers
   • Covers: USA, science, computing, math, etc.
    ↓ (no match)
3. Web Search + Scraping
   • DuckDuckGo search
   • Scrapes top 3 URLs
   • Extracts clean content
   • Filters ads/garbage
    ↓
4. Response Generation
   • RAG-style extraction
   • Quality validation
   • Iterative refinement
    ↓
5. Feedback Loop
   • "Was this helpful?"
   • Learn from positive feedback
   • Self-correct from negative feedback
```

### 2. Continuous Learning System

**Tracks Everything:**
- Every query and answer
- Which method provided the answer
- User feedback (positive/negative)
- Usage patterns and popular topics

**Learns Automatically:**
- Stores helpful answers in learned Q&A
- Builds personalized knowledge base
- Adapts to your domain over time

**Self-Corrects:**
- Removes bad answers when marked unhelpful
- Extracts corrections from your comments
- Learns corrected answers automatically

### 3. Web Scraping Integration

- Searches DuckDuckGo for relevant results
- Scrapes top 3 URLs with RobustWebScraper
- Respects robots.txt and rate limits
- Extracts clean content, filters noise
- Stores in knowledge base for future use
- Learns Q&A pairs from successful scraping

### 4. Negative Feedback Intelligence

**7 Natural Correction Patterns:**
1. "The X is Y"
2. "Correct answer is Y"
3. "It should be Y"
4. "Actually it's Y"
5. "X means Y"
6. "Should be Y"
7. Simple "is Y" patterns

**Example:**
```
You› what is the full form of USA

μSage› [Bad answer]

Was this helpful? (y/n/skip): n
What went wrong?: The correct answer is United States of America

[System automatically:]
• Removes bad answer from learned Q&A
• Extracts "United States of America"
• Learns corrected answer
• Next time: serves correct answer!
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    μSage Agent                          │
│  (Orchestrates all components)                          │
└─────────────────────────────────────────────────────────┘
            ↓           ↓           ↓           ↓
    ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Learning  │ │ Simple   │ │   Web    │ │Response  │
    │  System   │ │   Q&A    │ │ Scraper  │ │Generator │
    └───────────┘ └──────────┘ └──────────┘ └──────────┘
            ↓           ↓           ↓           ↓
    ┌───────────────────────────────────────────────────┐
    │          Knowledge Base + Embeddings              │
    │          (FAISS + all-MiniLM-L6-v2)               │
    └───────────────────────────────────────────────────┘
```

### Data Storage

```
~/.musage/
├── learning/
│   ├── learned_qa.json      # Your learned Q&A pairs
│   ├── usage_log.json       # Last 1000 queries
│   ├── feedback.json        # All feedback records
│   └── stats.json           # Usage statistics
├── knowledge_base.pkl       # Stored web knowledge
├── faiss_index.bin          # Semantic search index
└── conversation_history.pkl # Chat context
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Quick Install

```bash
# Install from source
cd project-musage
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Dependencies

**Core:**
- beautifulsoup4 - Web scraping
- requests - HTTP requests
- duckduckgo-search - Web search
- sentence-transformers - Embeddings
- faiss-cpu - Similarity search

**Optional:**
- transformers - For Qwen LLM (500MB)
- torch - For neural models

---

## Quick Start

### Basic Usage

```bash
# Run μSage
python -m musage

# Or if installed
musage
```

### Example Session

```
You› what is machine learning

μSage› Machine learning is a subset of artificial intelligence 
       where computers learn from data to make predictions 
       without being explicitly programmed.

Was this helpful? (y/n/skip): y

Thanks! μSage is learning from your feedback.

You› what is ML

μSage› Machine learning is a subset of artificial intelligence...
       [Retrieved from learned Q&A - <0.001s]
```

### CLI Commands

```bash
# Show learning statistics
learnstats

# Export learned knowledge
export

# Clear conversation history
clear

# Exit
exit
quit
bye
```

---

## Learning System

### How It Works

**1. Query Logging**
- Every query recorded with metadata
- Tracks: query, answer, method, timestamp
- Identifies popular topics

**2. Feedback Collection**
- Automatic prompt after each answer
- Records positive/negative feedback
- Collects optional comments

**3. Automatic Learning**
- Positive feedback → Learn Q&A pair
- Negative feedback → Remove bad answer + extract correction
- Confidence scoring (0.8 initial, adjusts with usage)

**4. Intelligent Retrieval**
- Exact match first
- Fuzzy matching (85%+ similarity)
- Confidence threshold filtering

### Statistics Dashboard

```bash
You› learnstats

╔══════════════════════════════════════════════════════════════╗
║           μSage Learning Statistics                          ║
╚══════════════════════════════════════════════════════════════╝

Usage:
   • Total queries: 247
   • Learned Q&A pairs: 45
   • First used: 2026-02-01
   • Last used: 2026-02-23

Answer Methods:
   • Learned Q&A: 112 queries (45%) ← INSTANT!
   • Simple Q&A: 48 queries (19%)
   • Web Scraping: 87 queries (35%)

User Feedback:
   • Positive: 198 (93% satisfaction)
   • Negative: 14
   • Self-corrections applied: 8

Top Topics:
   • machine learning: 15 queries
   • python programming: 12 queries
   • quantum computing: 8 queries
```

### Export Learned Knowledge

```bash
You› export

Exported 45 learned Q&A pairs to learned_qa.py

# File contents:
LEARNED_QA = {
    "machine learning": "Machine learning is...",
    "quantum computing": "Quantum computing is...",
    # ... more entries
}
```

---

## Negative Feedback & Self-Correction

### The Problem (Fixed!)

Previously, bad answers persisted even after marking unhelpful. User corrections were ignored.

### The Solution

**When you mark an answer unhelpful:**

1. **Bad answer removed** from learned Q&A
2. **Correction extracted** from your comment
3. **Corrected answer learned** automatically
4. **Next query** uses corrected answer

### Correction Patterns

The system understands natural language corrections:

```python
# Pattern examples that work:
"The full form of USA is United States of America"
"The correct answer is Paris"
"It should be approximately 299,792,458 m/s"
"Actually it's a systems programming language"
"AI means artificial intelligence"
"Should be more specific" ← Filtered (too vague)
"Should be Paris" ← Learned (specific)
```

### Example Workflow

```
Query: "what is the capital of france"
Answer: "London" (BAD!)

User marks: n
Comment: "The correct answer is Paris"

System:
  1. Remove "London" from learned Q&A
  2. Extract "Paris" from comment
  3. Learn new Q&A: "capital of france" → "Paris"
  4. Save to learned_qa.json

Next query: "capital of france"
Answer: "Paris" (corrected!)
```

---

## Answer Quality & Pipeline

### Quality Validation

Every answer goes through validation:

**1. Garbage Detection**
- Filters: ads, navigation menus, "sign up" prompts
- Removes: "image will be uploaded", newsletter prompts
- Blocks: off-topic or wrong-language content

**2. Relevance Scoring**
- Matches query keywords to content
- Scores sentence relevance
- Ranks: DEFINITION → ELABORATION → EXAMPLE

**3. Iterative Refinement**
- Initial extraction
- Quality evaluation (discriminator)
- Targeted refinement if needed (max 2 iterations)
- Threshold: 0.7 overall quality

### Answer Pipeline

```python
def generate_answer(query):
    # 1. Check learned Q&A (personalized)
    if learned_answer := learning.get_learned_answer(query):
        return learned_answer
    
    # 2. Check Simple Q&A (built-in)
    if simple_answer := simple_qa.get_answer(query):
        return simple_answer
    
    # 3. Web search + scraping
    results = web_search(query)
    content = scrape_urls(results[:3])
    
    # 4. Extract and validate
    answer = extract_answer(query, content)
    if not validate(answer):
        return honest_cant_answer()
    
    # 5. Log for learning
    learning.log_query(query, answer, "extraction")
    
    return answer
```

---

## CLI Commands

### Interactive Mode

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `learnstats` | Show learning statistics |
| `export` | Export learned Q&A to file |
| `clear` | Clear conversation history |
| `exit`, `quit`, `bye` | Exit μSage |

### Feedback Prompts

After each answer:
```
Was this helpful? (y/n/skip):
  y - Learn this answer
  n - Remove bad answer, optionally provide correction
  skip - Don't record feedback
```

If you choose `n`:
```
What went wrong? (optional):
  [Type your correction or explanation]
```

---

## API Reference

### ResponseGenerator

```python
from musage.responses import ResponseGenerator

generator = ResponseGenerator()

# Generate answer
answer = generator.generate_from_sources(
    query="what is python",
    sources=[...],
    context="",
    intent="definition"
)

# Record feedback
generator.record_feedback(
    query="what is python",
    answer="Python is...",
    helpful=True,
    comment=""
)

# Get statistics
stats = generator.get_stats()
```

### LearningSystem

```python
from musage.learning import get_learning_system

learning = get_learning_system()

# Log query
learning.log_query(
    query="what is AI",
    answer="AI is...",
    method="simple_qa",
    success=True
)

# Get learned answer
answer = learning.get_learned_answer("what is AI")

# Record feedback
learning.record_feedback(
    query="what is AI",
    answer="AI is...",
    helpful=True,
    comment=""
)

# Get statistics
summary = learning.get_stats_summary()
```

### Simple Q&A

```python
from musage.simple_qa import get_answer

# Get answer (with fuzzy matching)
answer = get_answer("what is usa", threshold=0.75)

# Check if answer exists
if answer:
    print(answer)  # "USA stands for United States of America..."
```

---

## Configuration

### Environment Variables

```bash
# Set custom storage directory
export MUSAGE_HOME=/path/to/storage

# Disable learning system
export MUSAGE_DISABLE_LEARNING=1
```

### In Code

```python
# musage/config.py

# Web search settings
MAX_SEARCH_RESULTS = 5
SCRAPER_TIMEOUT = 10
SCRAPER_MAX_WORKERS = 3

# Learning settings
LEARNED_QA_THRESHOLD = 0.75
FUZZY_MATCH_THRESHOLD = 0.85

# Response generation
LLM_MAX_TOKENS = 150
REFINEMENT_THRESHOLD = 0.7
```

---

## Development Guide

### Project Structure

```
project-musage/
├── musage/
│   ├── __init__.py
│   ├── agent.py          # Main orchestrator
│   ├── cli.py            # CLI interface
│   ├── responses.py      # Answer generation
│   ├── learning.py       # Continuous learning
│   ├── simple_qa.py      # Built-in Q&A
│   ├── search.py         # Web search
│   ├── scraper.py        # Web scraping
│   ├── knowledge.py      # Knowledge base
│   ├── embeddings.py     # Semantic search
│   └── conversation.py   # Context management
├── docs/
│   └── DOCUMENTATION.md  # This file
├── README.md
├── CHANGELOG.md
├── requirements.txt
└── setup.py
```

### Running Tests

```bash
# No test files in production
# Use interactive testing:
python -m musage
```

### Adding New Features

**1. Add to Simple Q&A:**
```python
# musage/simple_qa.py
QA_DATABASE = {
    "your new question": "your answer here",
    # ... more entries
}
```

**2. Modify Learning Patterns:**
```python
# musage/learning.py
def _extract_correction_from_comment(query, comment):
    # Add your pattern
    match = re.search(r'your_pattern', comment)
    if match:
        return match.group(1)
```

**3. Extend Response Generation:**
```python
# musage/responses.py
def generate_from_sources(...):
    # Add your logic
    pass
```

---

## Troubleshooting

### Common Issues

**1. No internet connection**
```
Error: "I'm not able to reach the internet"
Solution: Check network. System will fall back to cached knowledge.
```

**2. Import errors**
```
Error: "No module named 'musage'"
Solution: pip install -e . or run from project root
```

**3. Slow responses**
```
Issue: First query takes long
Reason: Loading embeddings model (~90MB)
Solution: Normal behavior, subsequent queries faster
```

**4. Bad answers persist**
```
Issue: Same bad answer after marking unhelpful
Solution: Ensure you're providing correction in comment
         System needs clear correction like "The answer is X"
```

**5. Learning not working**
```
Issue: Answers not being learned
Check: ~/.musage/learning/ has write permissions
Check: Feedback was recorded (y/n, not skip)
```

### Debug Mode

```bash
# Enable logging
export MUSAGE_LOG_LEVEL=DEBUG
python -m musage
```

### Reset Learning Data

```bash
# Remove learned Q&A (keeps feedback logs)
rm ~/.musage/learning/learned_qa.json

# Complete reset
rm -rf ~/.musage/learning/
```

---

## Best Practices

### For Users

**Provide clear corrections:**
- "The correct answer is X"
- "It should be X"
- Not just "wrong" or "incorrect"

**Mark helpful answers:**
- Helps build your personalized knowledge
- Makes future queries faster

**Check statistics regularly:**
- Run `learnstats` to see progress
- Track satisfaction percentage

### For Developers

**Test in isolated environment:**
```bash
python -m venv test_env
source test_env/bin/activate
pip install -e .
```

**Respect data storage:**
- Don't modify JSON files directly
- Use API methods for changes

**Follow naming conventions:**
- Methods: `lowercase_with_underscores`
- Classes: `CamelCase`
- Constants: `UPPERCASE_WITH_UNDERSCORES`

---

## Performance

### Benchmarks

| Method | Response Time | First Query | Subsequent |
|--------|---------------|-------------|------------|
| Learned Q&A | <0.001s | - | <0.001s |
| Simple Q&A | <0.001s | <0.001s | <0.001s |
| Web Scraping | 3-5s | 3-5s | 3-5s (if not learned) |
| Extraction | 0.1-0.3s | 0.1-0.3s | 0.1-0.3s |

### Memory Usage

- Base system: ~50MB
- With embeddings: ~150MB
- Per learned Q&A: ~1KB
- 1000 queries logged: ~500KB

### Storage

- Learned Q&A (100 pairs): ~100KB
- Knowledge base (1000 entries): ~10MB
- FAISS index: ~5MB per 1000 embeddings

---

## Roadmap

### Completed
- Web scraping integration
- Continuous learning system
- Negative feedback intelligence
- Simple Q&A built-in answers
- Statistics dashboard
- Export functionality

### Planned
- Multi-language support
- Voice input/output
- Browser extension
- API server mode
- Collaborative learning (share learned Q&A)

---

## Credits

**Developed by:** [Md. Abid Hasan Rafi](https://abidhasanrafi.github.io/)  
**Powered by:** [AI Extension](https://aiextension.org/)  
**GitHub Repository:** [github.com/abidhasanrafi/muSage](https://github.com/abidhasanrafi/muSage)

**Key Technologies:**
- sentence-transformers (all-MiniLM-L6-v2)
- FAISS (Facebook AI Similarity Search)
- DuckDuckGo Search API
- BeautifulSoup4
- Python 3.8+

---

## License

See LICENSE file for details.

---

## Support

For issues, questions, or contributions:
- Check this documentation first
- Review CHANGELOG.md for recent changes
- Test in a clean environment
- Provide detailed error messages

---

## Version History

See CHANGELOG.md for complete version history.

**Current Version: 0.3.0**
- Continuous learning system
- Negative feedback intelligence
- Self-correction from user feedback
- 60+ built-in Q&A topics
- Statistics dashboard
- Export functionality

---

**Last Updated:** February 23, 2026  
**Documentation Version:** 1.0.0
