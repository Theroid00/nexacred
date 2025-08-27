#!/usr/bin/env python3
"""
Fast Financial RAG Chatbot with Conversation Memory
=================================================

A lightweight, fast-responding financial chatbot that prioritizes speed over complexity.
Uses pre-built responses and simple matching for instant answers with natural conversation flow
and conversation memory to remember previous interactions.
"""

import sys
import time
import re
import random
from typing import Dict, List, Tuple

from conversation_memory import ConversationMemory, get_conversation_memory

class FastFinancialChatbot:
    """Fast financial chatbot with pre-built knowledge base, natural responses, and conversation memory."""

    def __init__(self):
        """Initialize the chatbot with pre-built financial knowledge, conversational elements, and memory."""

        # Initialize conversation memory
        self.conversation_memory = get_conversation_memory()
        self.current_session_id = None

        # Conversational starters for more natural responses
        self.response_starters = [
            "Based on current Indian financial regulations, ",
            "According to RBI guidelines, ",
            "In the Indian financial system, ",
            "From what I understand about Indian banking, ",
            "Here's what you should know: ",
            "Let me explain this clearly: ",
            "To give you the most accurate information: ",
            "Based on the latest financial norms, "
        ]

        self.helpful_additions = [
            " Would you like me to explain any specific aspect in more detail?",
            " Is there anything specific about this you'd like to know more about?",
            " Let me know if you need clarification on any part of this.",
            " Feel free to ask if you have any follow-up questions.",
            " I hope this helps! Do you have any other questions?",
            " Would you like me to explain the process or requirements in more detail?"
        ]

        self.knowledge_base = {
            # RBI Guidelines - Enhanced with more natural language
            "rbi": {
                "capital adequacy": {
                    "response": "Banks in India are required to maintain a minimum Capital Adequacy Ratio (CAR) of 9% under Basel III guidelines. This ensures banks have sufficient capital to handle potential losses. The requirement includes maintaining Tier 1 capital of at least 6% and Common Equity Tier 1 (CET1) capital of at least 4.5%.",
                    "follow_up": "This regulation helps protect depositors and maintains financial stability."
                },
                "kyc": {
                    "response": "All banks must implement comprehensive Know Your Customer (KYC) procedures, which include customer identification, verification, and ongoing monitoring. Enhanced due diligence is required for high-risk customers. Digital KYC using Aadhaar authentication is now permitted with proper customer consent.",
                    "follow_up": "This helps prevent money laundering and ensures financial security."
                },
                "digital payments": {
                    "response": "Payment service providers must implement multi-factor authentication for all transactions above ₹5,000. Real-time fraud monitoring systems are mandatory, incorporating transaction velocity checks and behavioral analytics to detect suspicious activities.",
                    "follow_up": "These measures significantly enhance the security of digital transactions."
                },
                "personal loans": {
                    "response": "Personal loans are unsecured credit facilities offered by banks and NBFCs without requiring collateral. Interest rates typically range from 10-24% per annum, depending on your credit score and income profile. The maximum loan amount can go up to ₹40 lakhs, subject to your income and creditworthiness.",
                    "follow_up": "Your credit score plays a crucial role in determining both eligibility and interest rates."
                },
                "home loans": {
                    "response": "Home loans in India currently have interest rates ranging from 8-12% per annum, varying by lender and borrower profile. The maximum loan-to-value ratio is 90% for loans up to ₹30 lakhs and 80% for higher amounts. You can choose repayment tenures of up to 30 years.",
                    "follow_up": "Longer tenures reduce EMI but increase total interest paid over the loan period."
                }
            },

            # SEBI Guidelines - Enhanced
            "sebi": {
                "mutual funds": {
                    "response": "SEBI has established strict asset allocation norms for mutual funds to protect investors. Large cap funds must invest at least 80% in large cap stocks, while mid cap funds require 65% investment in mid cap stocks. Risk management includes mandatory daily NAV calculation and regular portfolio disclosure.",
                    "follow_up": "These regulations ensure transparency and help investors make informed decisions."
                },
                "p2p lending": {
                    "response": "Peer-to-peer lending platforms in India must maintain escrow accounts with scheduled commercial banks for investor protection. Individual borrowers have a maximum exposure limit of ₹50,000 across all P2P platforms combined. Platform operators must maintain a minimum net worth of ₹2 crore to ensure operational stability.",
                    "follow_up": "These regulations balance innovation with investor protection in the growing P2P sector."
                },
                "p2p lending legal": {
                    "response": "Yes, P2P lending is completely legal in India and is regulated by SEBI since 2017. P2P platforms must be registered with SEBI as Non-Banking Financial Companies (NBFCs). The regulatory framework ensures investor protection through mandatory escrow accounts, exposure limits, and compliance requirements.",
                    "follow_up": "This legal framework makes P2P lending a legitimate investment and borrowing option in India."
                },
                "p2p lending regulations": {
                    "response": "P2P lending in India is governed by comprehensive SEBI regulations that include mandatory platform registration, escrow account maintenance, borrower exposure limits of ₹50,000 across all platforms, minimum net worth requirements of ₹2 crore for operators, and regular compliance reporting.",
                    "follow_up": "These regulations ensure the sector operates safely within legal boundaries."
                },
                "securities": {
                    "response": "SEBI comprehensively regulates securities markets in India to ensure fair trading and investor protection. All investors must complete KYC procedures with SEBI-registered intermediaries before trading. For derivatives trading, additional risk profiling and income verification are mandatory.",
                    "follow_up": "This multi-layered approach helps maintain market integrity and protects retail investors."
                }
            },

            # Credit and Banking - Enhanced
            "credit": {
                "credit score": {
                    "response": "Credit scores in India range from 300 to 900, with scores above 750 considered excellent for loan approvals. India has four major credit bureaus: CIBIL, Experian, Equifax, and CRIF High Mark. Each bureau may have slightly different scoring models, but all follow similar principles.",
                    "follow_up": "Regularly checking your credit report from all bureaus helps maintain a healthy credit profile."
                },
                "credit cards": {
                    "response": "Credit cards in India typically charge interest rates of 2-4% per month (equivalent to 24-48% annually) on outstanding balances. The minimum payment is usually 5% of the total outstanding amount. Late payment charges can range from ₹500 to ₹1,500 depending on the outstanding amount.",
                    "follow_up": "Paying the full amount by the due date helps avoid these high interest charges completely."
                },
                "loan eligibility": {
                    "response": "Loan eligibility assessment considers multiple factors: your credit score, monthly income, employment history, and existing debt-to-income ratio. Financial institutions generally prefer that your total EMIs don't exceed 40-50% of your monthly income to ensure comfortable repayment.",
                    "follow_up": "Maintaining a good credit score and stable income significantly improves your loan approval chances."
                }
            },

            # Digital Banking - Enhanced
            "digital": {
                "upi": {
                    "response": "UPI (Unified Payments Interface) enables instant money transfers 24x7, revolutionizing digital payments in India. Each bank account has a daily transaction limit of ₹1 lakh. Person-to-person transfers up to ₹1,000 are completely free of charge across all participating banks.",
                    "follow_up": "UPI has become the backbone of India's digital payment ecosystem due to its convenience and zero cost."
                },
                "neft rtgs": {
                    "response": "NEFT (National Electronic Funds Transfer) operates in hourly batches during banking hours, suitable for smaller amounts. RTGS (Real Time Gross Settlement) is designed for amounts above ₹2 lakhs and settles immediately in real-time. Both systems have specific operating windows and applicable charges.",
                    "follow_up": "Choose NEFT for smaller amounts and RTGS when you need immediate settlement of larger amounts."
                },
                "mobile banking": {
                    "response": "Mobile banking applications must implement robust two-factor authentication for security. Transaction limits are individually set by banks based on customer profiles and risk assessment. Many banks now use biometric authentication (fingerprint/face recognition) for enhanced security.",
                    "follow_up": "Always download banking apps from official app stores and enable all available security features."
                }
            }
        }

        # Enhanced quick responses with more context
        self.quick_responses = {
            "emi": {
                "response": "EMI (Equated Monthly Installment) is your fixed monthly payment towards a loan, which includes both principal repayment and interest charges. The EMI amount depends on loan amount, interest rate, and tenure.",
                "follow_up": "You can use online EMI calculators to determine the best loan structure for your budget."
            },
            "cibil": {
                "response": "CIBIL is India's first and largest credit information bureau. Your CIBIL score ranges from 300-900 and significantly impacts loan approvals, interest rates, and credit card eligibility.",
                "follow_up": "You can check your CIBIL score for free once a year directly from their website."
            },
            "roi": {
                "response": "ROI (Rate of Interest) represents the percentage charged on borrowed money or earned on investments, typically expressed as an annual percentage. It's a crucial factor in comparing different financial products.",
                "follow_up": "Always compare effective interest rates rather than just the headline rates when choosing loans."
            },
            "tenure": {
                "response": "Loan tenure refers to the total duration for repaying a loan, usually expressed in months or years. Longer tenures reduce EMI amounts but increase the total interest paid over the loan's lifetime.",
                "follow_up": "Choose a tenure that balances affordable EMIs with reasonable total interest costs."
            },
            "processing fee": {
                "response": "Processing fee is a one-time charge levied by lenders for evaluating and processing your loan application. It typically ranges from 0.5% to 2% of the loan amount and is usually non-refundable.",
                "follow_up": "Some lenders offer processing fee waivers during promotional periods, so it's worth asking."
            },
            "foreclosure": {
                "response": "Foreclosure allows you to repay your entire outstanding loan amount before the scheduled tenure ends. While this saves on future interest, some lenders charge a foreclosure penalty of 2-4% of the outstanding amount.",
                "follow_up": "Check your loan agreement for foreclosure terms before making early repayments."
            }
        }

    def find_best_match(self, query: str) -> Tuple[str, float, str, str]:
        """Find the best matching response for a query with improved scoring."""
        query_lower = query.lower()
        best_response = ""
        best_follow_up = ""
        best_score = 0.0
        source = ""

        # Check quick responses first (exact term matches)
        for term, content in self.quick_responses.items():
            if term in query_lower:
                if len(term) > best_score:
                    best_response = content["response"]
                    best_follow_up = content["follow_up"]
                    best_score = len(term) * 2  # Boost exact term matches
                    source = f"Financial Term: {term.upper()}"

        # Check knowledge base with improved matching
        for category, topics in self.knowledge_base.items():
            for topic, content in topics.items():
                # Calculate match score based on keyword presence
                keywords = topic.split()
                matches = 0
                total_keyword_length = 0

                for keyword in keywords:
                    if keyword in query_lower:
                        matches += 1
                        total_keyword_length += len(keyword)

                # Also check for category name
                category_match = 0
                if category in query_lower:
                    category_match = 1

                # Improved scoring
                if keywords:
                    keyword_coverage = matches / len(keywords)
                    specificity_bonus = total_keyword_length / 10
                    score = (keyword_coverage * 10) + specificity_bonus + (category_match * 2)

                    # Special handling for perfect matches
                    if matches == len(keywords) and len(keywords) > 1:
                        score += 5  # Bonus for complete topic match

                    if score > best_score:
                        best_response = content["response"]
                        best_follow_up = content["follow_up"]
                        best_score = score
                        source = f"{category.upper()}: {topic.title()}"

        return best_response, best_score, source, best_follow_up

    def generate_response(self, query: str, use_memory: bool = True) -> str:
        """Generate a natural, conversational response to the user's query with memory context."""

        # Get conversation context if memory is enabled and we have a session
        conversation_context = ""
        if use_memory and self.current_session_id:
            conversation_context = self.conversation_memory.get_conversation_context(
                self.current_session_id, max_messages=6  # Increased to get more context
            )

        # Add user message to memory
        if use_memory and self.current_session_id:
            self.conversation_memory.add_message(
                self.current_session_id,
                "user",
                query,
                metadata={"source": "fast_chatbot"}
            )

        # Enhanced query processing with better context awareness
        enhanced_query = query
        previous_topic = ""
        recent_user_topic = ""

        if conversation_context and "Previous conversation:" in conversation_context:
            # Extract both assistant responses and user questions for better context
            context_lines = conversation_context.split('\n')

            # Get the most recent topics discussed
            for line in reversed(context_lines):
                if line.startswith("Assistant:") and not previous_topic:
                    previous_topic = line.replace("Assistant:", "").strip()
                elif line.startswith("User:") and not recent_user_topic:
                    recent_user_topic = line.replace("User:", "").strip()

                if previous_topic and recent_user_topic:
                    break

            # Handle follow-up questions more intelligently
            query_lower = query.lower().strip()

            # Check for questions that need context from previous discussion
            context_questions = [
                "is it legal", "is this legal", "legal", "allowed", "permitted",
                "is it safe", "safe", "risky", "risks", "problems", "issues",
                "how does it work", "how it works", "process", "procedure",
                "what are the steps", "steps", "requirements", "eligibility",
                "benefits", "advantages", "disadvantages", "pros", "cons"
            ]

            needs_context = any(phrase in query_lower for phrase in context_questions)

            if needs_context and previous_topic:
                # Enhance the query with previous context
                if "p2p" in previous_topic.lower() or "peer-to-peer" in previous_topic.lower():
                    if "legal" in query_lower:
                        enhanced_query = "p2p lending legal regulations SEBI RBI compliance"
                    elif any(word in query_lower for word in ["safe", "risky", "risk"]):
                        enhanced_query = "p2p lending risks safety investor protection"
                    elif any(word in query_lower for word in ["work", "process", "steps"]):
                        enhanced_query = "p2p lending process detailed explanation"
                    else:
                        enhanced_query = f"p2p lending {query}"

                elif "credit score" in previous_topic.lower():
                    if any(word in query_lower for word in ["improve", "increase", "better"]):
                        enhanced_query = "credit score improvement methods detailed steps"
                    else:
                        enhanced_query = f"credit score {query}"

                elif "loan" in previous_topic.lower():
                    enhanced_query = f"loan {query}"

                elif "upi" in previous_topic.lower():
                    enhanced_query = f"upi {query}"

                # If we still don't have a good enhancement, use the recent user topic
                if enhanced_query == query and recent_user_topic:
                    enhanced_query = f"{query} about {recent_user_topic}"

            # Check for affirmative responses that want more detail
            elif query_lower in ["yes", "yeah", "ok", "okay", "sure", "tell me more", "more", "continue"]:
                # User wants more detail about the previous topic
                if "p2p" in previous_topic.lower() or "peer-to-peer" in previous_topic.lower():
                    enhanced_query = "p2p lending process requirements detailed explanation"
                elif "credit score" in previous_topic.lower():
                    enhanced_query = "credit score improvement methods detailed steps"
                elif "loan" in previous_topic.lower():
                    enhanced_query = "loan application process detailed requirements"
                elif "upi" in previous_topic.lower():
                    enhanced_query = "upi features benefits detailed explanation"
                else:
                    enhanced_query = f"detailed explanation of {previous_topic[:50]}"

            # Handle specific follow-up requests
            elif any(phrase in query_lower for phrase in ["more detail", "explain more", "tell me about", "how does", "what are the steps"]):
                if len(query.split()) < 4 and previous_topic:
                    # Short follow-up, enhance with previous topic
                    enhanced_query = f"{query} about {previous_topic[:50]}"

        response, score, source, follow_up = self.find_best_match(enhanced_query)

        # Check if we're repeating the same response
        is_repetition = False
        if conversation_context:
            context_lines = conversation_context.split('\n')
            for line in context_lines:
                if line.startswith("Assistant:") and response in line:
                    is_repetition = True
                    break

        if is_repetition and query.lower().strip() in ["yes", "yeah", "ok", "okay", "sure", "tell me more", "more"]:
            # User wants more detail but we're about to repeat - provide additional info
            response = self._get_detailed_followup(enhanced_query, previous_topic)
            score = 8.0  # High confidence for detailed response
            follow_up = "Would you like to know about any other aspect of this topic?"

        if score > 2.0:  # Good match found
            # Add conversational starter with context awareness
            if conversation_context and score > 5.0 and not is_repetition:
                context_starters = [
                    "Following up on our discussion, ",
                    "Building on what we talked about, ",
                    "To continue from where we left off, ",
                    "As a follow-up to your previous question, ",
                    "Expanding on that topic, "
                ]
                starter = random.choice(self.response_starters + context_starters)
            else:
                starter = random.choice(self.response_starters)

            # Construct natural response
            natural_response = f"{starter}{response}"

            # Add follow-up suggestion
            if follow_up:
                natural_response += f" {follow_up}"

            # Add helpful addition
            helpful = random.choice(self.helpful_additions)
            natural_response += helpful

        elif score > 0.5:  # Partial match
            starter = "I found some relevant information that might help: "
            natural_response = f"{starter}{response}"

            if follow_up:
                natural_response += f" {follow_up}"

            natural_response += " If this doesn't fully answer your question, please feel free to rephrase it or ask more specifically."

        else:  # No good match
            natural_response = self._generate_helpful_fallback(query, conversation_context)

        # Add assistant response to memory
        if use_memory and self.current_session_id:
            self.conversation_memory.add_message(
                self.current_session_id,
                "assistant",
                natural_response,
                metadata={"source": "fast_chatbot", "score": score}
            )

        return natural_response

    def _get_detailed_followup(self, enhanced_query: str, previous_topic: str) -> str:
        """Generate detailed follow-up information when user asks for more details."""

        # Detailed explanations for common topics
        detailed_responses = {
            "p2p": """Here's how P2P lending works in detail:

1. **Registration Process**: Both lenders and borrowers must register on SEBI-registered P2P platforms with KYC documentation.

2. **Risk Assessment**: Platforms conduct credit scoring and risk assessment for borrowers using algorithms and credit bureau data.

3. **Loan Listing**: Borrowers create loan requests with purpose, amount (max ₹10 lakhs), and proposed interest rates.

4. **Matching**: Lenders browse loan requests and choose to fund partially or fully based on risk appetite.

5. **Documentation**: Legal loan agreements are created automatically by the platform.

6. **Fund Transfer**: Money flows through escrow accounts maintained with scheduled commercial banks.

7. **Repayment**: Borrowers repay through EMIs, and platforms handle collection and distribution to lenders.

8. **Default Handling**: Platforms have recovery mechanisms, but lenders bear the credit risk.""",

            "credit score": """Here are detailed steps to improve your credit score:

1. **Payment History (35% impact)**:
   - Pay all EMIs and credit card bills on time
   - Set up auto-pay to avoid missed payments
   - Clear any overdue amounts immediately

2. **Credit Utilization (30% impact)**:
   - Keep credit card usage below 30% of limit
   - Ideally maintain below 10% for best scores
   - Spread purchases across multiple cards if needed

3. **Credit History Length (15% impact)**:
   - Keep old credit accounts open
   - Don't close your first credit card
   - Maintain long-term banking relationships

4. **Credit Mix (10% impact)**:
   - Have a healthy mix of secured and unsecured loans
   - Include credit cards, personal loans, and home loans

5. **New Credit Inquiries (10% impact)**:
   - Limit hard inquiries to 1-2 per year
   - Space out loan applications
   - Check your free credit report annually""",

            "upi": """UPI features and benefits in detail:

**Core Features:**
- 24x7 availability including weekends and holidays
- Instant money transfer between bank accounts
- Single mobile app for multiple bank accounts
- QR code based payments for merchants
- Bill payment and online shopping integration

**Security Features:**
- Two-factor authentication with UPI PIN
- Registered mobile number verification
- Bank-grade encryption for all transactions
- No sharing of bank account details with merchants

**Transaction Limits:**
- ₹1 lakh per day per bank account
- ₹5,000 per transaction without additional authentication
- Higher limits available with enhanced KYC

**Benefits:**
- Zero cost for P2P transfers up to ₹1,000
- Lower merchant transaction costs compared to cards
- Works across all participating banks
- Supports multiple languages
- Offline functionality in limited scenarios"""
        }

        # Try to find appropriate detailed response
        for key, detailed_response in detailed_responses.items():
            if key in enhanced_query.lower() or key in previous_topic.lower():
                return detailed_response

        # Default detailed response
        return f"""Let me provide more detailed information about {previous_topic[:50]}:

This is a comprehensive topic in Indian financial regulations. The key aspects include:

• **Regulatory Framework**: Governed by RBI/SEBI guidelines with specific compliance requirements
• **Implementation**: Step-by-step processes that institutions must follow
• **Consumer Protection**: Built-in safeguards to protect customer interests
• **Compliance Requirements**: Mandatory procedures and documentation
• **Recent Updates**: Latest changes in regulations and their implications

For specific details about any of these aspects, please ask about the particular area you're most interested in."""

    def _generate_helpful_fallback(self, query: str, conversation_context: str = "") -> str:
        """Generate a helpful response when no good match is found, considering conversation context."""

        # Context-aware fallback responses
        if conversation_context:
            context_fallbacks = [
                "I understand you're asking about something related to our previous discussion, but I need a bit more clarity. Could you rephrase your question or be more specific about what you'd like to know?",

                "Based on our conversation so far, I can see you're interested in financial topics, but I need more details about your current question. Could you elaborate on what specific aspect you're asking about?",

                "I want to make sure I give you the right information that builds on what we've discussed. Could you clarify what specific financial topic or regulation you're asking about?",
            ]
            response = random.choice(context_fallbacks)
        else:
            fallback_responses = [
                "I'd love to help you with that financial question, but I need a bit more context. Could you rephrase your question or be more specific about what you'd like to know?",

                "I don't have specific information about that topic in my current knowledge base. However, I can help you with questions about RBI guidelines, SEBI regulations, credit scores, loans, digital payments, and general banking queries.",

                "That's an interesting question! While I don't have detailed information on that specific topic, I'm well-versed in Indian financial regulations, banking procedures, and common financial products. Could you ask me something more specific in those areas?",

                "I want to make sure I give you accurate information. Could you clarify what specific aspect of Indian finance or banking you're asking about? I'm particularly knowledgeable about RBI guidelines, credit systems, loans, and digital payments."
            ]
            response = random.choice(fallback_responses)

        # Add topic suggestions based on query keywords
        suggestions = []
        query_lower = query.lower()

        if any(word in query_lower for word in ["loan", "credit", "borrow"]):
            suggestions.append("loan eligibility criteria")
        if any(word in query_lower for word in ["payment", "digital", "upi"]):
            suggestions.append("digital payment systems")
        if any(word in query_lower for word in ["bank", "account"]):
            suggestions.append("banking regulations")
        if any(word in query_lower for word in ["invest", "mutual", "fund"]):
            suggestions.append("investment guidelines")

        if suggestions:
            response += f"\n\nFor example, I can help with: {', '.join(suggestions)}."

        return response

    def resume_or_start_conversation(self, user_id: str = "fast_chatbot_user") -> str:
        """Resume the most recent conversation or start a new one."""
        # Try to find the most recent conversation for this user
        recent_sessions = self.conversation_memory.get_user_conversations(user_id, limit=1)

        if recent_sessions:
            # Resume the most recent conversation
            self.current_session_id = recent_sessions[0]
            # Load the conversation to make sure it's in memory
            self.conversation_memory._load_conversation(self.current_session_id)
            print(f"💭 Resuming previous conversation: {self.current_session_id[:8]}...")

            # Show a brief summary of what was discussed
            history = self.get_conversation_history()
            if history:
                last_topic = ""
                for msg in reversed(history):
                    if msg["role"] == "user" and len(msg["content"]) > 3:
                        last_topic = msg["content"]
                        break
                if last_topic:
                    print(f"📝 We were discussing: {last_topic[:50]}...")

            return self.current_session_id
        else:
            # Start a new conversation
            return self.start_new_conversation(user_id)

    def start_new_conversation(self, user_id: str = "fast_chatbot_user") -> str:
        """Start a completely new conversation session."""
        self.current_session_id = self.conversation_memory.start_conversation(user_id)
        return self.current_session_id

    def get_conversation_history(self) -> List[Dict]:
        """Get the current conversation history."""
        if not self.current_session_id:
            return []

        messages = self.conversation_memory.get_conversation_history(self.current_session_id)
        return [{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp} for msg in messages]

    def clear_conversation(self) -> bool:
        """Clear the current conversation."""
        if self.current_session_id:
            success = self.conversation_memory.clear_conversation(self.current_session_id)
            if success:
                self.current_session_id = None
            return success
        return True

    def chat(self):
        """Interactive chat interface with improved user experience and conversation memory."""
        print("🏦 NexaCred Fast Financial Assistant")
        print("=" * 50)
        print("Hello! I'm your fast financial assistant, specialized in Indian banking and finance.")
        print("I can quickly answer questions about:")
        print("• RBI and SEBI guidelines")
        print("• Loans and credit systems")
        print("• Digital payments and banking")
        print("• Credit scores and financial terms")
        print("\nI'll remember our conversation to provide better context-aware responses!")
        print("\nCommands:")
        print("• Type 'quit', 'exit', or 'bye' to end our conversation")
        print("• Type 'clear' to start a fresh conversation")
        print("• Type 'history' to see our conversation")
        print("=" * 50)

        # Resume previous conversation or start new one
        self.resume_or_start_conversation()

        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()

                if not user_input:
                    print("Please ask me a financial question, or type 'quit' to exit.")
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thank you for using NexaCred Financial Assistant!")
                    print("Have a great day and remember to stay financially informed! 💰")
                    break

                if user_input.lower() == 'clear':
                    self.clear_conversation()
                    self.start_new_conversation()
                    print(f"✨ Started fresh conversation: {self.current_session_id[:8]}...")
                    continue

                if user_input.lower() == 'history':
                    history = self.get_conversation_history()
                    if history:
                        print("\n📝 Conversation History:")
                        for i, msg in enumerate(history[-6:], 1):  # Show last 6 messages
                            role_emoji = "💬" if msg["role"] == "user" else "🤖"
                            print(f"{i}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}...")
                    else:
                        print("📝 No conversation history yet.")
                    continue

                # Simulate brief processing time for natural feel
                print("🤔 Let me check that for you...")
                time.sleep(0.3)

                start_time = time.time()
                response = self.generate_response(user_input, use_memory=True)
                response_time = time.time() - start_time

                print(f"\n🤖 Assistant: {response}")
                print(f"\n⚡ Response time: {response_time:.2f}s")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using NexaCred Financial Assistant!")
                break
            except Exception as e:
                print(f"\n❌ I encountered an error: {e}")
                print("Let's try again with your question.")

if __name__ == "__main__":
    chatbot = FastFinancialChatbot()
    chatbot.chat()
