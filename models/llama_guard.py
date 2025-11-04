# models/llama_guard.py
# Simple rule-based input filter simulating LlamaGuard behavior.
# Replace with real model-based guard later.

#import re
#from src.defenses import contextual_denial

#SENSITIVE_PATTERNS = [
 #   r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
  #  r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
   # r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
   # r"\bwhy are\b.*(bad at|worse at|less intelligent)",
   # r"\bgive me\b.*(phone numbers|email addresses|home address)"
#]

#def check_input(prompt: str):
 #   low = prompt.lower()
  #  for pat in SENSITIVE_PATTERNS:
   #     if re.search(pat, low):
    #        return contextual_denial(reason="input_pattern_match", alternative="I can provide safe, non-actionable information instead.")
    # If you later integrate LlamaGuard model, call it here and return contextual_denial on unsafe verdicts.
    #return {"allowed": True}



# models/llama_guard.py
"""
Input guard using meta-llama/Llama-Guard-4-12B.
Performs regex checks first (cheap), then calls the guard model (FP16, device_map="auto").
Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
"""

import os
import re
import time
from typing import Tuple
from src.defenses import contextual_denial
from src.constant import CFG

# simple regex patterns (cheap first-line defense)
SENSITIVE_PATTERNS = [
    r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
    r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
    r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
    r"\bwhy are\b.*(bad at|worse at|less intelligent)",
    r"\bgive me\b.*(phone numbers|email addresses|home address)",
    r"\b(list|give|show)\b.*\b(reasons|ways)\b.*\b(people|those|people from)\b"
]

# lazy-loaded guard pipeline
_guard_pipeline = None

def _load_guard():
    """
    Load the Llama-Guard model into GPU (FP16). Uses device_map='auto'.
    Returns a transformers pipeline or None on failure.
    """
    global _guard_pipeline
    if _guard_pipeline is not None:
        return _guard_pipeline

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        model_name = CFG.get("guard", {}).get("guard_model", "meta-llama/Llama-Guard-4-12B")
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": "torch.float16"
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        _guard_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, do_sample=False)
        print(f"[LlamaGuard] Loaded {model_name} (FP16, device_map=auto)")
    except Exception as e:
        print("[LlamaGuard] Failed to load guard model:", e)
        _guard_pipeline = None
    return _guard_pipeline

def _regex_check(prompt: str) -> bool:
    low = prompt.lower()
    for p in SENSITIVE_PATTERNS:
        if re.search(p, low):
            return True
    return False

def _guard_classify(prompt: str, threshold: float = None) -> Tuple[bool, float, str]:
    """
    Call the guard model and heuristically interpret the generated text.
    Returns: (flagged_bool, score_float, raw_model_text)
    Score is a 0..1-like value derived from presence of refusal keywords.
    """
    pipeline_guard = _load_guard()
    if pipeline_guard is None:
        # If guard model unavailable, do not flag here (regex already ran)
        return False, 0.0, ""

    try:
        # Some guard models use instruction wrappers; we send prompt directly here.
        out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
        text = out.lower()
        unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]
        score = 0.0
        for w in unsafe_words:
            if w in text:
                score += 1.0
        score = score / max(1, len(unsafe_words))
        guard_thresh = threshold if threshold is not None else CFG.get("guard", {}).get("guard_threshold", 0.55)
        flagged = score >= guard_thresh
        return flagged, float(score), out
    except Exception as e:
        print("[LlamaGuard] Classification error:", e)
        return False, 0.0, ""

def check_input(prompt: str):
    """
    Public function used by the pipeline.
    1) Run fast regex checks (if enabled in config)
    2) Run model-based guard (Llama-Guard) and return contextual denial if flagged
    """
    # 1) quick regex blocking (optional)
    if CFG.get("guard", {}).get("input_regex_block", True):
        if _regex_check(prompt):
            return contextual_denial(reason="input_pattern_match", alternative="I can provide safe, non-actionable information instead.")

    # 2) model-based guard
    flagged, score, raw = _guard_classify(prompt)
    if flagged:
        # include score info in logs via contextual_denial(reason=...) or separate logging if desired
        return contextual_denial(reason="input_guard_model", alternative="I can provide safe, non-actionable information instead.")

    # allowed
    return {"allowed": True}
