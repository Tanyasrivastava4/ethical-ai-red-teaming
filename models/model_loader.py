"""
models/model_loader.py
----------------------
This script loads and runs the LLM model (Mistral-7B) locally using Hugging Face.
You can later swap the model to LLaMA, Falcon, etc.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# You can change this if you want a smaller model later (like "TinyLlama" or "mistralai/Mistral-7B-Instruct-v0.2")
MODEL_NAME = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# -------------------------------
# 1️⃣ Load model and tokenizer
# -------------------------------
def load_model():
    print(f"🚀 Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",                     # automatically uses your GPU
        torch_dtype=torch.float16,             # half precision for efficiency
        load_in_8bit=True,                     # saves VRAM (optional)
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    print("✅ Model loaded successfully!\n")
    return generator


# -------------------------------
# 2️⃣ Generate model response
# -------------------------------
def generate_model_response(prompt, generator=None):
    """
    Takes a text prompt and returns model's generated response.
    """
    if generator is None:
        generator = load_model()

    result = generator(prompt, max_new_tokens=200)
    response = result[0]["generated_text"]

    # Sometimes the model repeats the prompt — we remove it
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response
