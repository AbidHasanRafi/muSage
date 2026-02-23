"""CLI interface for μSage (MuSage)"""

import sys
import time
import threading
import argparse
import logging
import textwrap
from colorama import init, Fore, Style

from .agent import MuSageAgent
from . import __version__, __author__, __powered_by__

# Initialize colorama for Windows support
init(autoreset=True)

# Only show WARNING+ from this module; verbose logging is silenced in __main__.py
logger = logging.getLogger(__name__)


# ── Typewriter helper ────────────────────────────────────────────────────────
def _typewrite(text: str, color: str = Fore.WHITE, delay: float = 0.013, end: str = '\n'):
    """Print text with a typewriter effect, one character at a time."""
    # Auto-reduce speed for long texts so it never feels sluggish
    if len(text) > 400:
        delay = 0.005
    elif len(text) > 200:
        delay = 0.009
    sys.stdout.write(color)
    sys.stdout.flush()
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(Style.RESET_ALL + end)
    sys.stdout.flush()


# ── Spinner ──────────────────────────────────────────────────────────────────
class _Spinner:
    """Animated braille spinner that runs in a background thread."""
    _FRAMES = ('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')

    def __init__(self, message: str, color: str = Fore.YELLOW):
        self.message = message
        self.color   = color
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(
                f"\r{self.color}  {frame}  {self.message}{Style.RESET_ALL}   "
            )
            sys.stdout.flush()
            time.sleep(0.09)
            i += 1

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 14) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


class MuSageCLI:
    """Interactive CLI for μSage (MuSage) Agent"""

    def __init__(self):
        self.agent = None
        self.running = False
        self.last_query = None
        self.last_answer = None
        self.answers_since_feedback = 0
        self.feedback_interval = 5  # Ask for feedback every 5 web answers

    def print_banner(self):
        """Print welcome banner with a subtle cascade-reveal effect."""
        W = 62

        def _row(label: str, value: str, vcol: str) -> str:
            inner = f"  {Fore.WHITE}{label}{vcol}{value}"
            pad   = W - 2 - len(label) - len(value)
            return f"{Fore.MAGENTA}║{inner}{' ' * max(pad, 0)}{Fore.MAGENTA}║{Style.RESET_ALL}"

        title_text = '·  μ S a g e  ·'
        sub_text   = 'Adaptive Web-Powered Conversational Assistant'

        lines = [
            "",
            f"{Fore.MAGENTA}╔{'═' * W}╗{Style.RESET_ALL}",
            f"{Fore.MAGENTA}║{' ' * W}║{Style.RESET_ALL}",
            f"{Fore.MAGENTA}║{Fore.CYAN + Style.BRIGHT}{title_text:^{W}}{Style.RESET_ALL}{Fore.MAGENTA}║{Style.RESET_ALL}",
            f"{Fore.MAGENTA}║{Fore.YELLOW}{sub_text:^{W}}{Style.RESET_ALL}{Fore.MAGENTA}║{Style.RESET_ALL}",
            f"{Fore.MAGENTA}║{' ' * W}║{Style.RESET_ALL}",
            f"{Fore.MAGENTA}║{'─' * W}║{Style.RESET_ALL}",
            f"{Fore.MAGENTA}║{' ' * W}║{Style.RESET_ALL}",
            _row("Developed by  : ", __author__,    Fore.GREEN),
            _row("Powered by    : ", __powered_by__, Fore.CYAN),
            _row("Version       : ", f"v{__version__}", Fore.WHITE),
            f"{Fore.MAGENTA}║{' ' * W}║{Style.RESET_ALL}",
            f"{Fore.MAGENTA}╚{'═' * W}╝{Style.RESET_ALL}",
            "",
        ]

        for line in lines:
            print(line)
            time.sleep(0.030)   # subtle cascade reveal

    def print_help(self):
        """Print help with a light typewriter intro."""
        bar = f"{Fore.CYAN}{'─' * 60}{Style.RESET_ALL}"
        print(f"\n{bar}")
        _typewrite("  Commands", Fore.CYAN + Style.BRIGHT, delay=0.035)
        print(bar)

        for cmd, desc in [
            ("help",       "Show this help message"),
            ("stats",      "Show knowledge base statistics"),
            ("learnstats", "Show learning & feedback statistics"),
            ("export",     "Export learned Q&A pairs"),
            ("version",    "Show version and credits"),
            ("clear",      "Clear all stored knowledge and conversations"),
            ("quit",       "Exit the application"),
        ]:
            print(f"  {Fore.GREEN}{cmd:<10}{Style.RESET_ALL}{Fore.WHITE}{desc}{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN + Style.BRIGHT}  Usage{Style.RESET_ALL}")
        print(f"  Type your question and press Enter.")
        print(f"  μSage searches the web and remembers answers for next time.\n")
        print(f"{Fore.CYAN + Style.BRIGHT}  Examples{Style.RESET_ALL}")
        for ex in [
            "What is quantum computing?",
            "How do neural networks work?",
            "Latest news on climate change?",
        ]:
            time.sleep(0.060)
            print(f"  {Fore.YELLOW}›{Style.RESET_ALL} {ex}")
        print(f"{bar}\n")

    def print_thinking(self, mode: str = 'search') -> _Spinner:
        """Start and return an animated spinner for the given mode."""
        cfg = {
            'search': ("Searching…",  Fore.YELLOW),
            'recall': ("Thinking…",  Fore.CYAN),
            'think':  ("Thinking…",  Fore.CYAN),
        }
        msg, color = cfg.get(mode, cfg['search'])
        return _Spinner(msg, color).start()

    def print_response(self, text: str, source: str = ''):
        """Print response with typewriter body."""
        sep   = f"{Fore.GREEN}{'─' * 62}{Style.RESET_ALL}"
        label = f"{Fore.GREEN + Style.BRIGHT}  μSage{Style.RESET_ALL}"
        print(f"\n{sep}")
        print(label)
        print(sep)

        for raw_line in text.splitlines():
            chunks = textwrap.wrap(raw_line, width=88) if len(raw_line) > 88 else [raw_line]
            for line in chunks:
                stripped = line.strip()
                if not stripped or stripped.startswith('[') or set(stripped) <= set('─—-='):
                    print(f"{Fore.WHITE}{line}{Style.RESET_ALL}")
                else:
                    _typewrite(line, Fore.WHITE, delay=0.013)

        print(f"{sep}")
        
        # Smart feedback: only ask occasionally for web answers, never for instant answers
        self._prompt_feedback_smart()
        print()

    def print_error(self, error: str):
        """Print error message."""
        print(f"\n{Fore.RED}  ✗  {error}{Style.RESET_ALL}\n")

    def get_input(self) -> str:
        """Get user input with styled prompt."""
        try:
            prompt = (
                f"{Fore.LIGHTMAGENTA_EX}  ╰─{Style.RESET_ALL}"
                f"{Fore.LIGHTMAGENTA_EX + Style.BRIGHT} You {Style.RESET_ALL}"
                f"{Fore.LIGHTMAGENTA_EX}›{Style.RESET_ALL} "
            )
            return input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            return "quit"

    def _prompt_feedback_smart(self):
        """Smart feedback: only ask occasionally for web answers, never for instant answers"""
        if not self.last_query or not self.last_answer or not self.agent:
            return
        
        # Get the source of the last answer
        source = getattr(self.agent, 'last_source', 'unknown')
        
        # Never ask for feedback on instant answers (they're already correct!)
        instant_sources = {'learned', 'simple_qa', 'builtin', 'local', 'conversational', 'memory'}
        if source in instant_sources:
            return
        
        # For web answers, only ask occasionally (every N answers)
        if source == 'web':
            self.answers_since_feedback += 1
            if self.answers_since_feedback < self.feedback_interval:
                return
            self.answers_since_feedback = 0
        
        # Prompt for feedback
        try:
            print(f"\n{Fore.CYAN}  Was this answer helpful? (y/n/skip): {Style.RESET_ALL}", end='')
            feedback = input().strip().lower()
            
            if feedback in ('y', 'yes'):
                print(f"{Fore.GREEN}  Thanks! \u03bcSage is learning from your feedback.{Style.RESET_ALL}")
                try:
                    self.agent.response_generator.learning.record_feedback(self.last_query, self.last_answer, True)
                except:
                    pass
            elif feedback in ('n', 'no'):
                print(f"{Fore.YELLOW}  Sorry about that. What went wrong? (optional): {Style.RESET_ALL}", end='')
                comment = input().strip()
                try:
                    self.agent.response_generator.learning.record_feedback(self.last_query, self.last_answer, False, comment)
                except:
                    pass
                print(f"{Fore.GREEN}  Feedback recorded. \u03bcSage will try to improve!{Style.RESET_ALL}")
        except:
            pass  # Skip feedback on any error

    def initialize_agent(self):
        """Initialize the agent with an animated loading spinner."""
        print()
        sp = _Spinner("Initializing μSage Agent…", Fore.YELLOW).start()
        try:
            self.agent = MuSageAgent()
            sp.stop()
            print(f"{Fore.GREEN}  ✓  Ready!{Style.RESET_ALL}\n")
            greeting = self.agent.get_greeting()
            self.print_response(greeting)
            return True
        except Exception as e:
            sp.stop()
            self.print_error(f"Failed to initialize: {e}")
            logger.exception("Initialization error")
            return False

    def handle_command(self, command: str):
        """
        Handle special commands.
        Returns True to continue, False to exit, None if not a command.
        """
        cmd = command.lower()

        if cmd in ('quit', 'exit', 'q'):
            _typewrite("\n  Thanks for using μSage! Goodbye! 👋", Fore.MAGENTA, delay=0.022)
            print()
            return False

        if cmd == 'help':
            self.print_help()
            return True

        if cmd == 'version':
            self.print_version()
            return True

        if cmd in ('stats', 'statistics', 'info'):
            sp = self.print_thinking('think')
            response = self.agent.ask('stats')
            sp.stop()
            self.print_response(response)
            return True

        if cmd in ('learnstats', 'learning', 'ls'):
            # Show learning system statistics
            try:
                stats = self.agent.response_generator.learning.get_stats_summary()
                print(f"\n{stats}\n")
            except Exception as e:
                print(f"{Fore.YELLOW}  Learning statistics not available: {e}{Style.RESET_ALL}\n")
            return True

        if cmd in ('export', 'exportlearned'):
            # Export learned Q&A
            try:
                filename = "learned_qa_export.py"
                self.agent.response_generator.learning.export_learned_qa(filename)
                print(f"{Fore.GREEN}  ✓  Learned Q&A exported to {filename}{Style.RESET_ALL}\n")
            except Exception as e:
                print(f"{Fore.YELLOW}  Export failed: {e}{Style.RESET_ALL}\n")
            return True

        if cmd == 'clear':
            confirm = input(
                f"{Fore.YELLOW}  Are you sure you want to clear all memory? (yes/no): {Style.RESET_ALL}"
            ).lower()
            if confirm == 'yes':
                self.agent.clear_memory()
                print(f"{Fore.GREEN}  ✓  All memory cleared{Style.RESET_ALL}\n")
            else:
                print(f"{Fore.CYAN}  Cancelled.{Style.RESET_ALL}\n")
            return True

        return None  # Not a command

    def run(self):
        """Main CLI loop."""
        self.print_banner()
        self.print_help()

        if not self.initialize_agent():
            return

        self.running = True

        while self.running:
            try:
                user_input = self.get_input()
                if not user_input:
                    continue

                result = self.handle_command(user_input)
                if result is False:
                    break
                if result is True:
                    continue

                # Regular question — show spinner until agent responds
                self.last_query = user_input
                sp = self.print_thinking('search')
                try:
                    response = self.agent.ask(user_input)
                    self.last_answer = response
                    sp.stop()
                    self.print_response(response)
                except Exception as e:
                    sp.stop()
                    self.print_error(f"Failed to process query: {e}")
                    logger.exception("Query error")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}  Use 'quit' to exit.{Style.RESET_ALL}\n")
            except Exception as e:
                self.print_error(f"Unexpected error: {e}")
                logger.exception("Unexpected error in main loop")

    def print_version(self):
        """Print version and credits."""
        print()
        _typewrite(f"  μSage (MuSage) v{__version__}", Fore.CYAN + Style.BRIGHT, delay=0.020)
        print(f"  {Fore.GREEN}Developer : {__author__}{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Powered by: {__powered_by__}{Style.RESET_ALL}")
        print()


def main():
    """Main entry point — supports --version, --about, and interactive mode"""
    from . import __version__, __author__, __powered_by__

    parser = argparse.ArgumentParser(
        prog="musage",
        description="μSage (MuSage) — Adaptive Web-Powered Conversational Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Developed by: {__author__}\n"
            f"Powered by:   {__powered_by__}\n"
            f"Version:      {__version__}"
        ),
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=(
            f"{Fore.CYAN}μSage (MuSage) v{__version__}{Style.RESET_ALL}\n"
            f"{Fore.GREEN}Developed by: {__author__}{Style.RESET_ALL}\n"
            f"{Fore.BLUE}Powered by:   {__powered_by__}{Style.RESET_ALL}"
        ),
    )
    parser.add_argument(
        "--about",
        action="store_true",
        help="Show detailed about information and exit",
    )

    args = parser.parse_args()

    if args.about:
        print(f"{Fore.CYAN}μSage (MuSage){Style.RESET_ALL}")
        print(f"  Small Machine Learning · Web-Augmented Intelligence")
        print(f"  {Fore.GREEN}Version   : {__version__}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Developer : {__author__}{Style.RESET_ALL}")
        print(f"  {Fore.BLUE}Powered by: {__powered_by__}{Style.RESET_ALL}")
        print(f"\n  Run {Fore.YELLOW}musage{Style.RESET_ALL} to start the interactive assistant.")
        return

    cli = MuSageCLI()
    try:
        cli.run()
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        logging.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
