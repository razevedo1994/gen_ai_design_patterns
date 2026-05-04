import os
from transformers import pipeline
import numpy as np

assert (
    os.environ["HF_TOKEN"][:2] == "hf"
), "Please sign up for access to the specific Llama model via HuggingFace and provide access token."


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


with open("banned_phrases.txt") as file:
    banned_phrases = [line.strip().lower() for line in file.readlines()]

with open("desired_phrases.txt") as file:
    desired_phrases = [line.strip().lower() for line in file.readlines()]

banned_phrases = set(banned_phrases)
desired_phrases = set(desired_phrases)


pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    use_fast=True,
    kwargs={
        "return_full_text": False,
    },
    model_kwargs={},
)


def generate_product_description(item: str) -> str:
    system_prompt = f"""
        You are a product marketer for a company that makes nutrition supplements.
        Balance your product descriptions to attract customers, optimize SEO, and
        stay within accurate advertising guidelines.
        Product descriptions have to be 3-5 sentences.
        Provide only the product description with no preamble.
    """

    user_prompt = f"""
        Write a product description for a {item}.
    """

    input_message = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    results = pipe(
        input_message,
        max_new_tokens=512,
    )

    return results[0]["generated_text"][-1]["content"].strip()


def evaluate(descr: str, positives, negatives) -> int:
    descr = descr.lower()

    num_positive = num_negative = 0
    for phrase in positives:
        if phrase in descr:
            num_positive += 1
            print(f"Good: {phrase}")
    for phrase in negatives:
        if phrase in descr:
            num_negative += 1
            print(f"Negative: {phrase}")
    print(f"Good counts {num_positive}\nBad counts: {num_negative}")
    return num_positive - num_negative


prod = generate_product_description("protein drink")
print(
    f"\n Product description (Zero-shot Generation): \n {'-' * 80}  \n {prod} \n {'-' * 80}"
)
evaluate(prod, desired_phrases, banned_phrases)
