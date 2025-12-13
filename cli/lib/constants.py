
def llm_prompt(context_texts: list[str], question: str):
    context = "\n".join(context_texts)
    prompt = f"""You are a helpful AI assistant.

Answer the user's question using only the information provided in the context below.
Do not use any outside knowledge.
If the answer is not clearly supported by the context, say that you do not know.

Context:
{context}

User question:
{question}
"""
    return prompt

