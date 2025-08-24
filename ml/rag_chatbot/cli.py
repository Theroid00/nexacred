"""
Command Line Interface
======================

CLI for the Indian Financial Regulation RAG Chatbot.
Provides interactive query interface for direct usage without web API.
"""

import asyncio
import logging
from typing import Optional
import sys
import argparse

from .config import Config
from .pipeline.rag import RAGPipeline
from .retrieval.dummy import DummyRetriever
from .retrieval.mongo_stub import MongoRetrieverStub
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


class ChatbotCLI:
    """Command line interface for the RAG chatbot."""
    
    def __init__(self, use_mongo: bool = False, config: Optional[Config] = None):
        """
        Initialize CLI with retriever choice.
        
        Args:
            use_mongo: Whether to use MongoDB retriever (stub)
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize retriever
        if use_mongo:
            logger.info("Using MongoDB retriever (stub implementation)")
            self.retriever = MongoRetrieverStub(self.config)
        else:
            logger.info("Using dummy retriever with sample data")
            self.retriever = DummyRetriever(self.config)
        
        # Initialize pipeline
        self.pipeline = RAGPipeline(self.retriever, self.config)
    
    def run_interactive(self):
        """Run interactive chat session."""
        print("\n🏦 Indian Financial Regulation RAG Chatbot")
        print("=" * 50)
        print("Ask questions about Indian financial regulations, loans, credit, debit cards, and P2P transactions.")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("Type 'help' for example questions.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                query = input("\n💬 Your question: ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thank you for using the chatbot.")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'health':
                    self._show_health()
                    continue
                
                # Process query
                print("\n🤔 Thinking...")
                result = self.pipeline.generate_response(query)
                
                # Display response
                self._display_response(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thank you for using the chatbot.")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                print(f"\n❌ Error: {str(e)}")
                print("Please try again with a different question.")
    
    def run_single_query(self, query: str):
        """
        Run a single query and return result.
        
        Args:
            query: User question
            
        Returns:
            Response dictionary
        """
        logger.info(f"Processing single query: {query}")
        return self.pipeline.generate_response(query)
    
    def _display_response(self, result: dict):
        """Display formatted response."""
        print("\n🤖 Response:")
        print("-" * 40)
        print(result["response"])
        
        # Show metadata if available
        metadata = result.get("metadata", {})
        if metadata:
            print(f"\n📊 Sources: {metadata.get('num_retrieved_docs', 0)} documents")
            
            if logger.isEnabledFor(logging.DEBUG):
                print(f"Context length: {metadata.get('context_length', 0)} chars")
                print(f"Status: {metadata.get('status', 'unknown')}")
        
        # Show retrieved documents if in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            retrieved_docs = result.get("retrieved_docs", [])
            if retrieved_docs:
                print("\n📚 Retrieved Documents:")
                for i, doc in enumerate(retrieved_docs[:2], 1):  # Show first 2
                    content = doc.get("content", "")[:100] + "..."
                    print(f"  {i}. {content}")
    
    def _show_help(self):
        """Show help with example questions."""
        print("\n❓ Example Questions:")
        print("-" * 30)
        examples = [
            "What are the eligibility criteria for personal loans in India?",
            "What are the RBI guidelines for peer-to-peer lending?",
            "What is the maximum interest rate for credit cards?",
            "How does the credit scoring system work in India?",
            "What are the regulations for digital payments?",
            "What are the KYC requirements for opening a bank account?",
            "What are the rules for foreign exchange transactions?",
            "How are cryptocurrency regulations structured in India?"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        
        print("\n💡 Tips:")
        print("  • Be specific in your questions")
        print("  • Ask about regulations, limits, processes, or requirements")
        print("  • Questions about loans, credit, payments, and banking are supported")
    
    def _show_health(self):
        """Show system health status."""
        print("\n🔍 System Health:")
        print("-" * 25)
        
        try:
            health = self.pipeline.health_check()
            for component, status in health.items():
                status_icon = "✅" if status in ["healthy", "loaded"] else "❌"
                print(f"  {status_icon} {component}: {status}")
        except Exception as e:
            print(f"  ❌ Health check failed: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Indian Financial Regulation RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode with dummy retriever
  %(prog)s --mongo                  # Interactive mode with MongoDB retriever
  %(prog)s --query "What are RBI guidelines for loans?"
  %(prog)s --debug --query "Credit card regulations"
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process (non-interactive mode)"
    )
    
    parser.add_argument(
        "--mongo", "-m",
        action="store_true",
        help="Use MongoDB retriever instead of dummy retriever"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else args.log_level
    setup_logging(level=log_level)
    
    try:
        # Initialize CLI
        cli = ChatbotCLI(use_mongo=args.mongo)
        
        if args.query:
            # Single query mode
            print(f"Processing query: {args.query}")
            result = cli.run_single_query(args.query)
            cli._display_response(result)
        else:
            # Interactive mode
            cli.run_interactive()
    
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"CLI error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
