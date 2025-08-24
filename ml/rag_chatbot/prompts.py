"""
Prompt Templates
================

System prompts and template builders for financial compliance queries.
"""


def build_prompt(context: str, user_query: str) -> str:
    """
    Build the exact prompt template as specified.
    
    Args:
        context: Retrieved context from regulation documents
        user_query: User's question
        
    Returns:
        Formatted prompt string for the model
    """
    
    prompt = f"""You are a financial compliance and credit regulation assistant.
Answer user questions strictly and only using the provided context from official banking rules, credit policies, and regulations.

Context:
{context}

User Question:
{user_query}

Instructions:
- Use only the information from the context.
- If the context does not contain the answer, reply exactly:
  "I cannot find this information in the current regulations database."
- Be precise, formal, and cite regulation titles/sections from the context when available.
- Do not invent information or make assumptions.

Answer:"""
    
    return prompt
