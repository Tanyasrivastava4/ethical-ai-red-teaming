#"""
#models/model_loader.py
#----------------------
#This script loads and runs the LLM model (Mistral-7B) locally using Hugging Face.
#You can later swap the model to LLaMA, Falcon, etc.
#"""

#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#import torch
#import os

# You can change this if you want a smaller model later (like "TinyLlama" or "mistralai/Mistral-7B-Instruct-v0.2")
MODEL_NAME = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# -------------------------------
# 1️⃣ Load model and tokenizer
# -------------------------------
#def load_model():
 #   print(f"🚀 Loading model: {MODEL_NAME}")

  #  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

   # model = AutoModelForCausalLM.from_pretrained(
    #    MODEL_NAME,
     #   device_map="auto",                     # automatically uses your GPU
      #  torch_dtype=torch.float16,             # half precision for efficiency
       # load_in_8bit=True,                     # saves VRAM (optional)
    #)

    #generator = pipeline(
     #   "text-generation",
      #  model=model,
      #  tokenizer=tokenizer,
      #  max_new_tokens=200,
      #  temperature=0.7,
      #  do_sample=True
    #)

    #print("✅ Model loaded successfully!\n")
    #return generator


# -------------------------------
# 2️⃣ Generate model response
# -------------------------------
#def generate_model_response(prompt, generator=None):
 #   """
  #  Takes a text prompt and returns model's generated response.
   # """
   # if generator is None:
    #    generator = load_model()

    #result = generator(prompt, max_new_tokens=200)
    #response = result[0]["generated_text"]

    # Sometimes the model repeats the prompt — we remove it
    #if response.startswith(prompt):
     #   response = response[len(prompt):].strip()

    #return response





# models/model_loader.py
"""
Load and generate with Mistral-7B (or fallback small model).
Provides generate(prompt) function used by red_team.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.constants import CFG

MODEL_NAME = CFG.get("model", {}).get("llm_model", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_NEW_TOKENS = CFG.get("model", {}).get("max_new_tokens", 128)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_generator = None

def _load_model():
    global _generator
    if _generator is not None:
        return _generator
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # attempt efficient loading: float16 + device_map auto + load_in_8bit if needed
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        # If you have very limited VRAM, consider adding load_in_8bit=True and bitsandbytes installed
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
        except Exception:
            # fallback: load without dtype/device_map (slower, CPU)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        device_id = 0 if _device == "cuda" else -1
        _generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device_id)
        print(f"[Model Loader] Loaded {MODEL_NAME} on {_device}")
    except Exception as e:
        print("Error loading model:", e)
        _generator = None
    return _generator

def generate(prompt: str, max_new_tokens: int = None):
    """
    Public generate function used by the pipeline.
    """
    gen = _load_model()
    if gen is None:
        return "[MODEL_ERROR] model not available"
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    try:
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = out[0].get("generated_text", "")
        # Remove prompt echo if present
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
    except Exception as e:
        return f"[MODEL_ERROR] {e}"
