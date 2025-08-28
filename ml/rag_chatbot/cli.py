"""
Command Line Interface
======================

Simplified CLI for the NexaCred RAG Chatbot.
Provides basic command-line access to the chatbot functionality.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fast_chatbot import FastFinancialChatbot
from conversation_memory import ConversationMemory


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="NexaCred Financial Assistant CLI")
    parser.add_argument("--query", "-q", help="Single query mode")
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--clear", action="store_true", help="Clear conversation history")

    args = parser.parse_args()
    
    if args.health:
        print("✅ NexaCred Financial Assistant is ready!")
        return

    if args.clear:
        memory = ConversationMemory()
        # Clear all conversations for the CLI user
        sessions = memory.get_user_conversations("cli_user")
        for session in sessions:
            memory.clear_conversation(session)
        print("✅ Conversation history cleared!")
        return

    # Initialize chatbot
    chatbot = FastFinancialChatbot()

    if args.query:
        # Single query mode
        session_id = chatbot.start_new_conversation("cli_user")
        response = chatbot.generate_response(args.query)
        print(f"Query: {args.query}")
        print(f"Response: {response}")
    else:
        # Interactive mode
        chatbot.chat()


if __name__ == "__main__":
    main()
