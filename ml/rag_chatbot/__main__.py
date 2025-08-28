#!/usr/bin/env python3
"""
NexaCred RAG Chatbot Main Entry Point
=====================================

Provides access to both fast template responses and full RAG AI responses.
Users can choose between speed (fast mode) or comprehensive AI analysis (RAG mode).
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for the NexaCred RAG Chatbot."""
    print("🏦 Welcome to NexaCred Financial Assistant")
    print("=" * 50)
    print("Choose your interaction mode:")
    print("1. Fast Mode - Instant template-based responses")
    print("2. AI Mode - Full RAG with IBM Granite 3.3 2B model")
    print("3. Exit")
    print("=" * 50)

    while True:
        try:
            choice = input("\nSelect mode (1/2/3): ").strip()

            if choice == "1":
                print("\n🚀 Starting Fast Mode...")
                run_fast_mode()
                break
            elif choice == "2":
                print("\n🤖 Starting AI Mode...")
                run_ai_mode()
                break
            elif choice == "3":
                print("\n👋 Goodbye!")
                sys.exit(0)
            else:
                print("Please enter 1, 2, or 3.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def run_fast_mode():
    """Run the fast template-based chatbot."""
    try:
        from fast_chatbot import FastFinancialChatbot
        chatbot = FastFinancialChatbot()
        chatbot.chat()
    except ImportError as e:
        print(f"Error loading fast chatbot: {e}")
        print("Please ensure all files are in the correct location.")
    except Exception as e:
        print(f"Error in fast mode: {e}")

def run_ai_mode():
    """Run the full RAG AI chatbot."""
    try:
        # Check if dependencies are installed
        missing_deps = check_dependencies()
        if missing_deps:
            print("❌ Missing required dependencies:")
            for dep in missing_deps:
                print(f"   • {dep}")
            print("\nInstalling dependencies...")
            install_dependencies()

        # Import and run RAG system with fixed imports
        import sys
        from pathlib import Path

        # Add current directory to path for proper imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))

        from pipeline.rag import RAGPipeline
        from config import Config

        print("🔄 Initializing AI system...")
        config = Config()

        # Try to use MongoDB retriever, fallback to dummy if needed
        try:
            if config.use_mongodb:
                print("🔄 Connecting to MongoDB Atlas...")
                from retrieval.mongo_stub import MongoRetriever
                retriever = MongoRetriever(
                    mongodb_uri=config.mongodb_uri,
                    database_name=config.mongodb_database,
                    collection_name=config.mongodb_collection,
                    index_name=config.mongodb_index_name
                )
                print("✅ MongoDB Atlas connected successfully!")

                # Display collection stats
                stats = retriever.get_collection_stats()
                print(f"📊 Database: {stats.get('database_name')}")
                print(f"📁 Collection: {stats.get('collection_name')}")
                print(f"📄 Total documents: {stats.get('total_documents', 0)}")
                print(f"🔍 Documents with embeddings: {stats.get('documents_with_embeddings', 0)}")

            else:
                print("🔄 Using dummy retriever (MongoDB disabled in config)...")
                from retrieval.dummy import DummyRetriever
                retriever = DummyRetriever()

        except Exception as e:
            print(f"⚠️ MongoDB connection failed: {e}")
            print("🔄 Falling back to dummy retriever...")
            from retrieval.dummy import DummyRetriever
            retriever = DummyRetriever()

        rag_pipeline = RAGPipeline(retriever, config)

        print("✅ AI system ready!")
        run_ai_chat(rag_pipeline)

    except ImportError as e:
        print(f"❌ Failed to load AI system: {e}")
        print("The AI system needs the full model download. Using enhanced fast mode...")
        run_fast_mode()
    except Exception as e:
        print(f"❌ Error in AI mode: {e}")
        print("Falling back to enhanced fast mode...")
        run_fast_mode()

def check_dependencies():
    """Check for required dependencies."""
    required_packages = [
        "torch", "transformers", "sentence_transformers",
        "scikit-learn", "huggingface_hub", "pydantic", "pydantic_settings"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return missing

def install_dependencies():
    """Install required dependencies."""
    import subprocess

    packages = [
        "torch", "transformers", "sentence-transformers",
        "scikit-learn", "huggingface-hub", "pydantic", "pydantic-settings"
    ]

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + packages)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please install manually: pip install torch transformers sentence-transformers scikit-learn")

def run_ai_chat(rag_pipeline):
    """Run the AI chat interface with conversation memory."""
    print("\n🤖 NexaCred AI Assistant (IBM Granite 3.3 2B)")
    print("=" * 50)
    print("I'm powered by IBM Granite 3.3 2B and can provide detailed analysis.")
    print("I'll remember our conversation to give you better context-aware responses!")
    print("\nCommands:")
    print("• Type 'quit', 'exit', or 'bye' to end our conversation")
    print("• Type 'clear' to start a fresh conversation")
    print("• Type 'history' to see our conversation")
    print("=" * 50)

    # Start a conversation session
    session_id = rag_pipeline.start_conversation("ai_chatbot_user")
    print(f"💭 Started conversation session: {session_id[:8]}...")

    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()

            if not user_input:
                print("Please ask me a financial question, or type 'quit' to exit.")
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Thank you for using NexaCred AI Assistant!")
                break

            if user_input.lower() == 'clear':
                rag_pipeline.clear_conversation()
                session_id = rag_pipeline.start_conversation("ai_chatbot_user")
                print(f"✨ Started fresh conversation: {session_id[:8]}...")
                continue

            if user_input.lower() == 'history':
                history = rag_pipeline.get_conversation_history()
                if history:
                    print("\n📝 Conversation History:")
                    for i, msg in enumerate(history[-6:], 1):  # Show last 6 messages
                        role_emoji = "💬" if msg["role"] == "user" else "🤖"
                        timestamp = msg.get("timestamp", "")
                        print(f"{i}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}...")
                else:
                    print("📝 No conversation history yet.")
                continue

            print("🤔 Analyzing your question with AI...")

            try:
                # Use the RAG pipeline with conversation memory
                response = rag_pipeline.query(user_input, include_context=True)
                print(f"\n🤖 AI Assistant: {response}")

                # Show memory stats occasionally
                import random
                if random.random() < 0.2:  # 20% chance
                    stats = rag_pipeline.get_memory_stats()
                    if stats.get("current_sessions", 0) > 0:
                        print(f"💭 Memory: {stats.get('current_sessions', 0)} active sessions")

            except Exception as e:
                print(f"❌ AI processing failed: {e}")
                print("Let me try to help with a quick response...")

                # Fallback to fast mode for this query
                from fast_chatbot import FastFinancialChatbot
                fast_bot = FastFinancialChatbot()
                # Set the same session if possible
                fast_bot.current_session_id = session_id
                fallback_response = fast_bot.generate_response(user_input, use_memory=True)
                print(f"\n🚀 Quick Response: {fallback_response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using NexaCred AI Assistant!")
            break
        except Exception as e:
            print(f"\n❌ I encountered an error: {e}")
            print("Let's try again with your question.")

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
