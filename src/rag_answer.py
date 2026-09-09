import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from retrieve import load_resources, retrieve


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SOURCE_LIST_FILE = Path("rag/source_list.csv")

MAX_NEW_TOKENS = 180


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

def load_llm():

    print("Loading Qwen tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    print("Loading Qwen model...")

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


# --------------------------------------------------
# Build evidence context
# --------------------------------------------------

def build_context(results):

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
[EVIDENCE {rank}]
Source ID: {result['source_id']}
Chunk ID: {result['chunk_id']}
Similarity Score: {result['score']:.4f}

{result['text']}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Generate grounded answer
# --------------------------------------------------

def generate_answer(
    question,
    results,
    tokenizer,
    model
):

    context = build_context(results)

    system_prompt = """
You are a waste-management information assistant.

Answer the user's question using ONLY the evidence passages
provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the evidence does not contain enough information to answer
   the question, say exactly:

"The available sources do not provide sufficient information
to answer this question."

4. Cite supporting sources ONLY using the exact format [S1], [S2],
   [S3], [S4], or [S5].
5. Never cite chunk IDs such as S2_00012.
6. Never write "Evidence", "Evidences", "Source:", or other
   citation formats.
7. Place citations immediately after the statement they support.
8. If multiple sources support a statement, use multiple citations,
   for example [S2][S3].
9. Keep the answer concise and factual.
10. Only cite a source when its evidence actually supports the
    statement.
"""

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Write a concise, evidence-grounded answer with source citations.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )

    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids
    else:
        input_ids = inputs

    input_ids = input_ids.to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated_tokens = outputs[
        0,
        input_ids.shape[-1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("GROUNDED RAG ANSWER GENERATION")
    print("=" * 70)

    print("\nLoading retrieval resources...")

    retrieval_model, index, metadata = load_resources()

    print(
        f"Indexed chunks: {index.ntotal}"
    )

    tokenizer, llm = load_llm()

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("No question provided.")

        return

    print("\nRetrieving evidence...")

    results = retrieve(
        question,
        retrieval_model,
        index,
        metadata
    )

    print("\nRetrieved sources:")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. {result['source_id']} "
            f"(score={result['score']:.4f})"
        )

    print("\nGenerating grounded answer...")

    answer = generate_answer(
        question,
        results,
        tokenizer,
        llm
    )

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)

    print(answer)


if __name__ == "__main__":
    main()