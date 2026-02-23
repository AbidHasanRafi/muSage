"""
Simple usage example for μSage (MuSage)

This demonstrates basic usage of the μSage agent.
Run this after installation to see μSage in action.
"""

from musage import MuSageAgent


def main():
    print("=" * 60)
    print("μSage (MuSage) Simple Example")
    print("=" * 60)
    print()
    
    # Initialize agent
    print("Initializing μSage Agent...")
    print("(First run may take a moment to download the embedding model)")
    print()
    
    agent = MuSageAgent()
    
    print("✓ Agent initialized!\n")
    print("=" * 60)
    
    # Example 1: Ask a question
    print("\n📝 Example 1: Asking a question\n")
    
    question1 = "What is Python programming language?"
    print(f"Question: {question1}")
    print("Searching and learning...\n")
    
    answer1 = agent.ask(question1)
    print(f"Answer:\n{answer1}\n")
    
    # Example 2: Ask a follow-up (uses context)
    print("=" * 60)
    print("\n📝 Example 2: Follow-up question\n")
    
    question2 = "What are its main features?"
    print(f"Question: {question2}")
    print("Searching and learning...\n")
    
    answer2 = agent.ask(question2)
    print(f"Answer:\n{answer2}\n")
    
    # Example 3: Check statistics
    print("=" * 60)
    print("\n📊 Example 3: View statistics\n")
    
    stats = agent.ask("stats")
    print(stats)
    
    # Show knowledge base info
    print("\n" + "=" * 60)
    print("\n✅ Examples complete!")
    print("\nμSage has learned from the web and stored knowledge locally.")
    print("Try asking the same questions again - they'll be answered instantly")
    print("from memory without needing to search the web!")
    
    print("\n💡 To run the full CLI: musage  (or: python -m musage)")
    print("=" * 60)


if __name__ == "__main__":
    main()
