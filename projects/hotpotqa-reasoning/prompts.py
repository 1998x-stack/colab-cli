"""CoT and ReAct prompt templates for HotpotQA reasoning comparison."""

COT_SYSTEM = (
    "You are a precise reasoning assistant. "
    "Answer the question using only the provided context. "
    "Think step by step, then give your final answer after 'Final Answer:'."
)

COT_TEMPLATE = """Context:
{context}

Question: {question}

Let's think step by step:

Step 1:"""

REACT_SYSTEM = (
    "You are a precise reasoning assistant. "
    "Answer the question using only the provided context. "
    "Use the ReAct format: Thought, then Action, then Observation. "
    "Actions should be Search[<thing to look up>] to find information in the context. "
    "When you have enough information, output 'Final Answer: <answer>'."
)

REACT_TEMPLATE = """Context:
{context}

Question: {question}

Thought:"""

REACT_CONTINUE_TEMPLATE = """Context:
{context}

Question: {question}

{history}
Thought:"""


def make_cot_prompt(question: str, context: str) -> str:
    return COT_TEMPLATE.format(context=context, question=question)


def make_react_initial_prompt(question: str, context: str) -> str:
    return REACT_TEMPLATE.format(context=context, question=question)


def make_react_continue_prompt(question: str, context: str, history: str) -> str:
    return REACT_CONTINUE_TEMPLATE.format(
        context=context, question=question, history=history
    )


def extract_final_answer(text: str) -> str | None:
    """Extract text after 'Final Answer:' marker. Returns None if not found."""
    marker = "Final Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    answer = text[idx + len(marker):].strip()
    return answer or None
