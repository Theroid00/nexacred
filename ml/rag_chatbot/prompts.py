"""
Prompt Templates
================

System prompts and template builders for financial compliance queries.
Optimized for IBM Granite 3.3 2B Instruct model.
"""


GRANITE_FINANCIAL_PROMPT = """<|system|>
You are a knowledgeable financial compliance assistant specializing in Indian banking regulations, credit policies, and financial guidelines. Provide accurate, helpful responses based strictly on the provided regulatory context.

<|user|>
Context:
{context}

Question: {question}

<|assistant|>
Based on the provided regulatory context, I can help you with this question about Indian financial regulations.

"""


def build_prompt(context: str, user_query: str) -> str:
    """
    Build the exact prompt template optimized for IBM Granite 3.3 2B Instruct.

    Args:
        context: Retrieved context from regulation documents
        user_query: User's question
        
    Returns:
        Formatted prompt string for the model
    """
    
    prompt = f"""<|system|>
You are a financial compliance and credit regulation assistant specializing in Indian financial regulations.
Answer questions strictly using the provided context from official banking rules, credit policies, and regulations.

<|user|>
Context:
{context}

Question:
{user_query}

Please provide a clear, accurate answer based only on the information in the context above.

<|assistant|>
Based on the regulatory context provided, """

    return prompt


# Alternative simpler prompt for fallback models
SIMPLE_FINANCIAL_PROMPT = """You are a financial compliance assistant for Indian regulations.

Context:
{context}

Question: {question}

Answer based on the context provided:"""


def get_prompt_for_model(model_id: str) -> str:
    """
    Get the appropriate prompt template based on the model being used.

    Args:
        model_id: The model identifier

    Returns:
        Appropriate prompt template
    """
    if "granite" in model_id.lower():
        return GRANITE_FINANCIAL_PROMPT
    else:
        return SIMPLE_FINANCIAL_PROMPT
