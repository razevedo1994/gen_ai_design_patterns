import os

assert os.environ["HF_TOKEN"][:2] == "hf", (
    "Please sign up for access to the specific Llama model via HuggingFace and provide access token."
)


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
