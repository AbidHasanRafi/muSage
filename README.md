<p align="center">
  <img src="assets/musage.jpg" alt="μSage Cover" width="800">
</p>

# μSage (MuSage) - Intelligent Self-Learning Assistant

**Continuously Learning · Web-Powered · Self-Correcting**

μSage is an intelligent assistant that learns from every interaction. It searches the web when needed, learns from your feedback, and builds a personalized knowledge base that gets faster and smarter over time.

---

## Key Features

- **Continuous Learning** - Learns Q&A pairs from feedback, builds personalized knowledge
- **Self-Correction** - Removes bad answers, extracts corrections, learns fixed answers  
- **Web Integration** - Searches DuckDuckGo, scrapes content, filters garbage
- **Gets Faster** - Learned queries answered instantly (<0.001s)
- **Statistics** - Track learning progress, satisfaction rates, top topics
- **Works Offline** - Falls back to cached knowledge + 60+ built-in Q&A
- **Improves Daily** - Week 1: 10% instant → Month 3: 70% instant!

---

## Quick Start

### Installation

```bash
cd project-musage
pip install -r requirements.txt
```

### Run μSage

```bash
python -m musage
```

### Example Session

```
You› what is machine learning

μSage› Machine learning is a subset of artificial intelligence 
       where computers learn from data to make predictions...

Was this helpful? (y/n/skip): y
✓ Learned! Next time I'll answer instantly.

You› what is ML

μSage› [Retrieved from learned Q&A - <0.001s]
       Machine learning is a subset of artificial intelligence...
```

---

## How It Works

```
User Query
    ↓
1. Learned Q&A (personalized, instant)
    ↓ (no match)
2. Simple Q&A (60+ built-in topics)
    ↓ (no match)
3. Web Search + Scraping
    ↓
4. Extract & Validate
    ↓
5. Feedback: "Was this helpful?"
   YES → Learn Q&A pair
   NO  → Remove bad answer, extract correction
    ↓
Gets Smarter!
```

---

## Learning Features

### Positive Feedback
- Mark helpful → System learns
- Next query → Instant retrieval
- Builds your domain knowledge

### Negative Feedback (Self-Correction)
- Mark unhelpful → Bad answer removed
- Provide correction → System learns fixed answer
- Supports natural language:
  - "The correct answer is X"
  - "It should be X"
  - "Actually it's X"

### Statistics Dashboard

```bash
You› learnstats

Usage: 247 queries, 45 learned, 93% satisfaction
Methods: 45% instant (learned), 19% built-in, 35% web
Topics: machine learning (15), python (12), quantum (8)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `learnstats` | Show learning statistics |
| `export` | Export learned Q&A to file |
| `clear` | Clear conversation history |
| `help` | Show help |
| `exit` | Exit μSage |

---

## Complete Documentation

**See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for:**
- Complete feature guide
- API reference
- Development guide  
- Troubleshooting
- Performance benchmarks
- Configuration options

---

## Dependencies

- `beautifulsoup4` - Web scraping
- `requests` - HTTP requests
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Similarity search
- `duckduckgo-search` - Web search

**Install:** `pip install -r requirements.txt`

---

## Project Structure

```
muSage/
├── musage/              # Core package
│   ├── agent.py        # Main orchestrator
│   ├── cli.py          # CLI interface
│   ├── responses.py    # Answer generation
│   ├── learning.py     # Continuous learning
│   ├── simple_qa.py    # Built-in Q&A
│   ├── search.py       # Web search
│   ├── scraper.py      # Web scraping
│   └── ...
├── docs/
│   └── DOCUMENTATION.md # Complete docs
├── README.md            # This file
├── requirements.txt     # Dependencies
└── setup.py            # Package setup
```

---

## How Learning Works

### Day 1
- Ask 50 questions
- All from web (3-5s each)
- Mark 20 helpful
- 20 Q&A learned

### Week 2
- Ask 50 questions
- 20 instant (learned)
- 10 built-in
- 20 new (web)
- 40% faster!

### Month 3
- Ask 50 questions
- 35 instant (learned)
- 10 built-in
- 5 new (web)
- 70% instant!

---

## Configuration

Configuration in `musage/config.py`:

```python
# Storage
BASE_DIR = Path.home() / ".musage"

# Learning
LEARNED_QA_THRESHOLD = 0.75
FUZZY_MATCH_THRESHOLD = 0.85

# Web scraping
MAX_SEARCH_RESULTS = 5
SCRAPER_TIMEOUT = 10
```

---

## Example Usage

### As Python Package

```python
from musage import MuSageAgent

agent = MuSageAgent()
response = agent.ask("What is Python?")
print(response)
```

### With Feedback

```python
# Record feedback
agent.record_feedback(
    query="What is Python?",
    answer=response,
    helpful=True
)

# Next time: instant retrieval!
```

---

## Important Notes

### Ethics
- Respects robots.txt
- Rate limiting built-in
- Caches aggressively

### Privacy
- All data local (~/.musage/)
- No telemetry
- No external services (except web search)

### Limitations
- Retrieval system, not generative AI
- Quality depends on web sources
- Requires internet for new queries

---

## Troubleshooting

### Slow First Run
- Downloads embedding model (~90MB)
- Normal behavior

### Bad Answer Persists
- Mark unhelpful (n)
- Provide correction in comment
- System will learn fixed answer

### Import Errors
```bash
pip install -e .
# Or run from project root
```

---

## Version

**Current:** v0.3.0  
**Features:** Continuous learning, self-correction, web scraping integration

See [`CHANGELOG.md`](CHANGELOG.md) for version history.

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Credits

**Developed by:** [Md. Abid Hasan Rafi](https://abidhasanrafi.github.io/)  
**Powered by:** [AI Extension](https://aiextension.org/)  
**GitHub:** [github.com/abidhasanrafi/muSage](https://github.com/abidhasanrafi/muSage)

**Technologies:**
- sentence-transformers (all-MiniLM-L6-v2)
- FAISS (Facebook AI)
- DuckDuckGo Search API
- BeautifulSoup4

---

## Quick Links

- **GitHub Repository:** [github.com/abidhasanrafi/muSage](https://github.com/abidhasanrafi/muSage)
- **Documentation:** [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md)
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)
- **Example:** [`example_usage.py`](example_usage.py)

---

**Built for continuous learning and improvement**

*μSage - Gets smarter with every question*
