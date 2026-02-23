# Changelog

All notable changes to μSage (MuSage) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-02-23

### Added
- Initial release of μSage (MuSage)
- Web search integration using DuckDuckGo API
- Robust web scraping with BeautifulSoup
  - robots.txt compliance
  - Rate limiting
  - Retry logic with exponential backoff
  - Intelligent content extraction
- Local knowledge base with persistent storage
- Lightweight embeddings using all-MiniLM-L6-v2 model
- FAISS integration for fast similarity search (with NumPy fallback)
- Conversation memory and context management
- CLI interface with colorful output
- Offline mode support using cached knowledge
- Extractive response generation
- Knowledge base statistics tracking
- Usefulness scoring system
- Commands: help, stats, clear, quit/exit
- Comprehensive documentation
  - README.md with full documentation
  - QUICKSTART.md for new users
  - EXAMPLES.md with code examples
  - DEVELOPMENT.md for contributors
- Test installation script
- Example usage script
- MIT License

### Features
- CPU-friendly design (no GPU required)
- Minimal resource usage (~500MB RAM)
- Fast cached responses (<1 second)
- Automatic online/offline detection
- Cross-platform support (Windows, Linux, macOS)
- Package distribution via pip
- Modular architecture for easy extension

### Dependencies
- beautifulsoup4 >= 4.12.0
- requests >= 2.31.0
- lxml >= 4.9.0
- sentence-transformers >= 2.2.0
- numpy >= 1.24.0
- faiss-cpu >= 1.7.4
- duckduckgo-search >= 3.9.0
- colorama >= 0.4.6
- tqdm >= 4.66.0

---

## [Unreleased]

### Planned for v0.2.0
- Conversation summarization
- Better context retention
- Knowledge base pruning (remove old entries)
- Multi-turn reasoning improvements
- Export/import knowledge base
- Web UI (optional)

### Planned for v0.3.0
- Local LLM integration (TinyLlama, Phi-2)
- Better extractive QA
- Multi-language support
- PDF/document parsing
- User feedback loop
- Custom search engines
- Plugin system

---

## Version History

- **0.1.0** (2024-02-23): Initial release

---

## Migration Guide

### From v0.0.x to v0.1.0
N/A - First release

---

## Breaking Changes

N/A - First release

---

## Contributors

See GitHub contributors page for full list.

---

## Support

For support, please:
1. Check the documentation (README.md, QUICKSTART.md)
2. Search existing issues
3. Create a new issue if needed
