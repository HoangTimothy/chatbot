REFUSAL_MESSAGE = "I cannot find sufficient information in the knowledge base."

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer only using the retrieved context.
If the context does not contain enough evidence, respond exactly:
I cannot find sufficient information in the knowledge base.
Do not invent policies, procedures, specifications, or facts."""


def build_context_block(chunks: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{chunk_id}]\n{text}" for chunk_id, text in chunks)

