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
#"""
#Input guard using meta-llama/Llama-Guard-4-12B.
#Performs regex checks first (cheap), then calls the guard model (FP16, device_map="auto").
#Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
#"""

#import os
#import re
#import torch
#import time
#from typing import Tuple
#from src.defenses import contextual_denial
#from src.constant import CFG

# simple regex patterns (cheap first-line defense)
#SENSITIVE_PATTERNS = [
 #   r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
  #  r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
  #  r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
  #  r"\bwhy are\b.*(bad at|worse at|less intelligent)",
  #  r"\bgive me\b.*(phone numbers|email addresses|home address)",
  #  r"\b(list|give|show)\b.*\b(reasons|ways)\b.*\b(people|those|people from)\b"
#]

# lazy-loaded guard pipeline
#_guard_pipeline = None

#def _load_guard():
 #   """
  #  Load the Llama-Guard model into GPU (FP16). Uses device_map='auto'.
   # Returns a transformers pipeline or None on failure.
   # """
   # global _guard_pipeline
   # if _guard_pipeline is not None:
    #    return _guard_pipeline

    #try:
     #   from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
      #  model_name = CFG.get("guard", {}).get("guard_model", "meta-llama/Llama-Guard-4-12B")
       # load_kwargs = {
        #    "device_map": "auto",
         #   "dtype": torch.float16
        #}
        #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        #model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        #_guard_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, do_sample=False)
        #print(f"[LlamaGuard] Loaded {model_name} (FP16, device_map=auto)")
    #except Exception as e:
     #   print("[LlamaGuard] Failed to load guard model:", e)
      #  _guard_pipeline = None
    #return _guard_pipeline

#def _regex_check(prompt: str) -> bool:
 #   low = prompt.lower()
  #  for p in SENSITIVE_PATTERNS:
   #     if re.search(p, low):
    #        return True
    #return False

#def _guard_classify(prompt: str, threshold: float = None) -> Tuple[bool, float, str]:
 #   """
  #  Call the guard model and heuristically interpret the generated text.
   # Returns: (flagged_bool, score_float, raw_model_text)
   # Score is a 0..1-like value derived from presence of refusal keywords.
   # """
   # pipeline_guard = _load_guard()
   # if pipeline_guard is None:
        # If guard model unavailable, do not flag here (regex already ran)
   #     return False, 0.0, ""

   # try:
        # Some guard models use instruction wrappers; we send prompt directly here.
    #    out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
    #    text = out.lower()
    #    unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]
     #   score = 0.0
     #   for w in unsafe_words:
      #      if w in text:
       #         score += 1.0
       # score = score / max(1, len(unsafe_words))
       # #guard_thresh = threshold if threshold is not None else CFG.get("guard", {}).get("guard_threshold", 0.55)
        #flagged = score >= guard_thresh
       # guard_thresh = threshold if isinstance(threshold, (float, int)) else CFG.get("guard", {}).get("guard_threshold", 0.55)
       # if guard_thresh is None:
        #    guard_thresh = 0.55  # default safety threshold
       # flagged = float(score) >= float(guard_thresh)

        
        #return flagged, float(score), out

    #try:
        # Some guard models use instruction wrappers; we send prompt directly here.
     #   out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
      #  text = out.lower()

       # unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]
       # score = 0.0
      #  for w in unsafe_words:
      #      if w in text:
       #         score += 1.0
      #  score = score / max(1, len(unsafe_words))

        #Get guard threshold safely
     #   cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)
    #    guard_thresh = threshold if isinstance(threshold, (float, int)) else cfg_thresh

    # ✅ Ensure both are valid floats
    #    try:
    #        guard_thresh = float(guard_thresh)
    #    except (TypeError, ValueError):
    #        guard_thresh = 0.55  # fallback default

    #    try:
     #      score = float(score)
     #   except (TypeError, ValueError):
     #       score = 0.0

     #   flagged = score >= guard_thresh

      #  return flagged, score, out

   # except Exception as e:
    #    print("[LlamaGuard] Classification error:", e)
     #   return False, 0.0, ""



 
    #except Exception as e:
     #   print("[LlamaGuard] Classification error:", e)
      #  return False, 0.0, ""

#def check_input(prompt: str):
 #   """
  #  Public function used by the pipeline.
   # 1) Run fast regex checks (if enabled in config)
    #2) Run model-based guard (Llama-Guard) and return contextual denial if flagged
   # """
    # 1) quick regex blocking (optional)
  #  if CFG.get("guard", {}).get("input_regex_block", True):
   #     if _regex_check(prompt):
    #        return contextual_denial(reason="input_pattern_match", alternative="I can provide safe, non-actionable information instead.")

    # 2) model-based guard
   # flagged, score, raw = _guard_classify(prompt)
   # if flagged:
        # include score info in logs via contextual_denial(reason=...) or separate logging if desired
   #     return contextual_denial(reason="input_guard_model", alternative="I can provide safe, non-actionable information instead.")

    # allowed
  #  return {"allowed": True}



#I was running this file and have error < None
#"""
#Input guard using meta-llama/Llama-Guard-4-12B.
#Performs regex checks first (cheap), then calls the guard model (FP16, device_map="auto").
#Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
#"""

#import os
#import re
#import torch
#import time
#from typing import Tuple
#from src.defenses import contextual_denial
#from src.constant import CFG

# simple regex patterns (cheap first-line defense)
#SENSITIVE_PATTERNS = [
 #   r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
  #  r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
  #  r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
  #  r"\bwhy are\b.*(bad at|worse at|less intelligent)",
  #  r"\bgive me\b.*(phone numbers|email addresses|home address)",
  #  r"\b(list|give|show)\b.*\b(reasons|ways)\b.*\b(people|those|people from)\b"
#]

# lazy-loaded guard pipeline
#_guard_pipeline = None


#def _load_guard():
 #   """
  #  Load the Llama-Guard model into GPU (FP16). Uses device_map='auto'.
   # Returns a transformers pipeline or None on failure.
    #"""
   # global _guard_pipeline
   # if _guard_pipeline is not None:
   #     return _guard_pipeline

   # try:
    #    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
     #   model_name = CFG.get("guard", {}).get("guard_model", "meta-llama/Llama-Guard-4-12B")

     #   load_kwargs = {
      #      "device_map": "auto",
       #     "torch_dtype": torch.float16
       # }

        #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
       # model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

#       # tokenizer.pad_token = tokenizer.eos_token_id

       # _guard_pipeline = pipeline(
        #    "text-generation",
         #   model=model,
          #  tokenizer=tokenizer,
           # max_new_tokens=64,
           # do_sample=False,
           # pad_token_id=tokenizer.eos_token_id
           #tokenizer.pad_token = tokenizer.eos_token_id

        #)
        #print(f"[LlamaGuard] Loaded {model_name} (FP16, device_map=auto)")
    #except Exception as e:
     #   print("[LlamaGuard] Failed to load guard model:", e)
      #  _guard_pipeline = None

    #return _guard_pipeline


#def _regex_check(prompt: str) -> bool:
 #   """Simple fast regex-based check before using the heavy model."""
  #  low = prompt.lower()
   # for p in SENSITIVE_PATTERNS:
    #    if re.search(p, low):
     #       return True
    #return False


#def _guard_classify(prompt: str, threshold: float = None) -> Tuple[bool, float, str]:
 #   """
  #  Call the guard model and heuristically interpret the generated text.
   # Returns: (flagged_bool, score_float, raw_model_text)
    #"""
   # pipeline_guard = _load_guard()
   # if pipeline_guard is None:
        # If guard model unavailable, do not flag here (regex already ran)
    #    return False, 0.0, ""

   # try:
        # Some guard models use instruction wrappers; we send prompt directly here.
    #    start_time = time.time()
     #   out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
      #  if time.time() - start_time > 20:
       #     print("[LlamaGuard] Warning: classification took >20s")

       # text = out.lower().strip()
       # unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]

        # calculate a simple heuristic score
        #score = sum(1.0 for w in unsafe_words if w in text) / max(1, len(unsafe_words))

        # safely retrieve threshold
        #cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)
        #guard_thresh = threshold if isinstance(threshold, (float, int)) else cfg_thresh

        # ensure both are floats
        #try:
         #   guard_thresh = float(guard_thresh)
        #except (TypeError, ValueError):
         #   guard_thresh = 0.55

        #try:
         #   score = float(score)
        #except (TypeError, ValueError):
         #   score = 0.0
        #score = float(score) if score is not None else 0.0
        #guard_thresh = float(guard_thresh) if guard_thresh is not None else 0.55

        # ensure both are floats
        #score = float(score) if score is not None else 0.0
       # guard_thresh = float(guard_thresh) if guard_thresh is not None else 0.55

        #flagged = score >= guard_thresh


        #flagged = score >= guard_thresh

        #return flagged, score, out

    #except Exception as e:
     #   print("[LlamaGuard] Classification error:", e)
      #  return False, 0.0, ""


#def check_input(prompt: str):
 #   """
  #  Public function used by the pipeline.
   # 1) Run fast regex checks (if enabled in config)
    #2) Run model-based guard (Llama-Guard) and return contextual denial if flagged
    #"""
    # 1) Quick regex blocking (optional)
    #if CFG.get("guard", {}).get("input_regex_block", True):
     #   if _regex_check(prompt):
      #      print("[LlamaGuard] Blocked by regex pattern match.")
       #     return contextual_denial(
        #        reason="input_pattern_match",
         #       alternative="I can provide safe, non-actionable information instead."
          #  )

    # 2) Model-based guard
   # flagged, score, raw = _guard_classify(prompt)
    #if flagged:
     #   print(f"[LlamaGuard] Blocked by model (score={score:.2f}).")
      #  return contextual_denial(
       #     reason="input_guard_model",
        #    alternative="I can provide safe, non-actionable information instead."
        #)

    # Allowed
    #return {"allowed": True}






"""
Input guard using meta-llama/Llama-Guard-4-12B.
Performs regex checks first (cheap), then calls the guard model (FP16, device_map="auto").
Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
"""

import os
import re
import torch
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
            "dtype": torch.float16
        }

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        _guard_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"[LlamaGuard] Loaded {model_name} (FP16, device_map=auto)")
    except Exception as e:
        print("[LlamaGuard] Failed to load guard model:", e)
        _guard_pipeline = None

    return _guard_pipeline


def _regex_check(prompt: str) -> bool:
    """Simple fast regex-based check before using the heavy model."""
    low = prompt.lower()
    for p in SENSITIVE_PATTERNS:
        if re.search(p, low):
            return True
    return False


def _guard_classify(prompt: str, threshold: float = None) -> Tuple[bool, float, str]:
    """
    Call the guard model and heuristically interpret the generated text.
    Returns: (flagged_bool, score_float, raw_model_text)
    """
    pipeline_guard = _load_guard()
    if pipeline_guard is None:
        # If guard model unavailable, do not flag here (regex already ran)
        return False, 0.0, ""

    try:
        # Some guard models use instruction wrappers; we send prompt directly here.
        start_time = time.time()
        out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
        if time.time() - start_time > 20:
            print("[LlamaGuard] Warning: classification took >20s")

        text = out.lower().strip()
        unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]

        # calculate a simple heuristic score
        score = sum(1.0 for w in unsafe_words if w in text) / max(1, len(unsafe_words))

        # safely retrieve threshold with proper defaults
        #cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)
        
        # Use provided threshold or fall back to config
        #if threshold is not None:
         #   try:
          #      guard_thresh = float(threshold)
           # except (TypeError, ValueError):
            #    guard_thresh = 0.55
        #else:
         #   try:
          #      guard_thresh = float(cfg_thresh)
           # except (TypeError, ValueError):
            #    guard_thresh = 0.55

        # Ensure score is a float (it should already be, but double-check)
        #score = float(score)

        # Compare and flag
        #print(f"[DEBUG] raw model output: '{out}'")
        #print(f"[DEBUG] text processed: '{text}'")
        #print(f"[DEBUG] score calculated: {score}")
        #print(f"[DEBUG] guard_thresh from config/arg: {guard_thresh}")

        #flagged = score >= guard_thresh

        # safely retrieve threshold with proper defaults
        cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)

# fallback logic made stricter
        if threshold is not None and isinstance(threshold, (int, float)):
            guard_thresh = float(threshold)
        else:
            guard_thresh = float(cfg_thresh) if isinstance(cfg_thresh, (int, float)) else 0.55

# Ensure score is always float and not None
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        # Debug prints
        print(f"[DEBUG] score: {score}, guard_thresh: {guard_thresh}")

        flagged = score >= guard_thresh

     
     
        #flagged = score >= guard_thresh

        return flagged, score, out

    except Exception as e:
        print("[LlamaGuard] Classification error:", e)
        return False, 0.0, ""


def check_input(prompt: str):
    """
    Public function used by the pipeline.
    1) Run fast regex checks (if enabled in config)
    2) Run model-based guard (Llama-Guard) and return contextual denial if flagged
    """
    # 1) Quick regex blocking (optional)
    if CFG.get("guard", {}).get("input_regex_block", True):
        if _regex_check(prompt):
            print("[LlamaGuard] Blocked by regex pattern match.")
            return contextual_denial(
                reason="input_pattern_match",
                alternative="I can provide safe, non-actionable information instead."
            )

    # 2) Model-based guard
    flagged, score, raw = _guard_classify(prompt)
    if flagged:
        print(f"[LlamaGuard] Blocked by model (score={score:.2f}).")
        return contextual_denial(
            reason="input_guard_model",
            alternative="I can provide safe, non-actionable information instead."
        )

    # Allowed
    return {"allowed": True}
