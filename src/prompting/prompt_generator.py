def build_basic_prompt(query, context):
    return f"""
You are a sentiment classification system.

Context:
{context}

Task:
Classify the sentiment of the following sentence as Positive or Negative.

Sentence: {query}

Answer:
"""


def build_context_prompt(query, retrieved_sentences):
    examples = "\n".join(retrieved_sentences)

    return f"""
You are an intelligent sentiment classifier.

Here are similar examples:
{examples}

Now classify the sentiment of:

"{query}"

Respond with only Positive or Negative.
"""


def build_cot_prompt(query, context):
    return f"""
You are a reasoning-based sentiment classifier.

Use step-by-step reasoning:

Context:
{context}

Sentence:
{query}

Step 1: Identify sentiment clues
Step 2: Compare with context examples
Step 3: Decide final sentiment

Final Answer:
"""