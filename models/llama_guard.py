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




# this code was before when i added llama-4-12B from huggingface architecture.

#"""
##Input guard using meta-llama/Llama-Guard-4-12B.
##Performs regex checks first (cheap), then calls the guard model (FP16, device_map="auto").
##Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
##"""

##import os
##import re
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
  #  Returns a transformers pipeline or None on failure.
  #  """
  #  global _guard_pipeline
  #  if _guard_pipeline is not None:
  #      return _guard_pipeline

#    try:
 #       from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
  #      model_name = CFG.get("guard", {}).get("guard_model", "meta-llama/Llama-Guard-4-12B")
#
 #       load_kwargs = {
  #          "device_map": "auto",
   #         "dtype": torch.float16
    #    }

    #    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    #    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    #    _guard_pipeline = pipeline(
     #       "text-generation",
     #       model=model,
     #       tokenizer=tokenizer,
     #       max_new_tokens=64,
     #       do_sample=False,
     #       pad_token_id=tokenizer.eos_token_id
     #   )
      #  print(f"[LlamaGuard] Loaded {model_name} (FP16, device_map=auto)")
    #except Exception as e:
     #   print("[LlamaGuard] Failed to load guard model:", e)
      #  _guard_pipeline = None

    #return _guard_pipeline


#def _regex_check(prompt: str) -> bool:
 #   """Simple fast regex-based check before using the heavy model."""
  #  low = prompt.lower()
   # for p in SENSITIVE_PATTERNS:
   #     if re.search(p, low):
    #        return True
   # return False


#def _guard_classify(prompt: str, threshold: float = None) -> Tuple[bool, float, str]:
 #   """
  #  Call the guard model and heuristically interpret the generated text.
   # Returns: (flagged_bool, score_float, raw_model_text)
   # """
   # pipeline_guard = _load_guard()
   # if pipeline_guard is None:
        # If guard model unavailable, do not flag here (regex already ran)
   #     return False, 0.0, ""

   # try:
        # Some guard models use instruction wrappers; we send prompt directly here.
    #    start_time = time.time()
     #   out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
      #  print(f"[DEBUG] Raw guard output:\n{out}\n")
#
 #       elapsed = time.time() - start_time
  #      if elapsed > 20:
   #         print(f"[LlamaGuard] Warning: classification took {elapsed:.1f}s")

    #    text = out.lower().strip()
     #   unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]

        # Calculate a simple heuristic score - ensure it's always a float
      #  score = float(sum(1.0 for w in unsafe_words if w in text) / max(1, len(unsafe_words)))

        # Safely retrieve threshold with proper defaults
       # cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)

        # Use provided threshold or fall back to config
       # if threshold is not None:
       #     try:
       #         guard_thresh = float(threshold)
        #    except (TypeError, ValueError):
         #       print("[LlamaGuard] Warning: Invalid threshold provided, using default 0.55")
          #      guard_thresh = 0.55
        #else:
         #   try:
          #      guard_thresh = float(cfg_thresh) if cfg_thresh is not None else 0.55
           # except (TypeError, ValueError):
            #    print("[LlamaGuard] Warning: Invalid config threshold, using default 0.55")
             #   guard_thresh = 0.55

        # Debug prints
        #print(f"[DEBUG BEFORE COMPARE] score={score} (type={type(score)}), guard_thresh={guard_thresh} (type={type(guard_thresh)})")

        # Safe comparison - both should be floats now
        #flagged = bool(score >= guard_thresh)

        #return flagged, score, out

    #except Exception as e:
     #   print(f"[LlamaGuard] Classification error: {e}")
      #  import traceback
       # traceback.print_exc()  # This will help debug the actual line causing the issue
        #return False, 0.0, ""



#def _guard_classify(prompt: str, threshold: float = None) -> Tuple[bool, float, str]:
 #   """
  #  Call the guard model and heuristically interpret the generated text.
  #  Returns: (flagged_bool, score_float, raw_model_text)
  #  """
  #  pipeline_guard = _load_guard()
  #  if pipeline_guard is None:
        # If guard model unavailable, do not flag here (regex already ran)
  #      return False, 0.0, ""

  #  try:
        # Some guard models use instruction wrappers; we send prompt directly here.
   #     start_time = time.time()
    #    out = pipeline_guard(prompt, max_new_tokens=64)[0].get("generated_text", "")
     #   print(f"[DEBUG] Raw guard output:\n{out}\n")

     #   if time.time() - start_time > 20:
      #      print("[LlamaGuard] Warning: classification took >20s")

      #  text = out.lower().strip()
      #  unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]

        # calculate a simple heuristic score
       # score = sum(1.0 for w in unsafe_words if w in text) / max(1, len(unsafe_words))

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
        #cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)

# fallback logic made stricter
        #if threshold is not None and isinstance(threshold, (int, float)):
         #   guard_thresh = float(threshold)
        #else:
         #   guard_thresh = float(cfg_thresh) if isinstance(cfg_thresh, (int, float)) else 0.55

# Ensure score is always float and not None
        #try:
         #   score = float(score)
        #except (TypeError, ValueError):
         #   score = 0.0

        # Debug prints
        #print(f"[DEBUG] score: {score}, guard_thresh: {guard_thresh}")
        #print(f"[DEBUG BEFORE COMPARE] score={score} (type={type(score)}), guard_thresh={guard_thresh} (type={type(guard_thresh)})")


        #flagged = score >= guard_thresh

     
     
        #flagged = score >= guard_thresh

        #return flagged, score, out

    #except Exception as e:
     #   print("[LlamaGuard] Classification error:", e)
      #  return False, 0.0, ""


#def check_input(prompt: str):
 #   """
  #  Public function used by the pipeline.
  #  1) Run fast regex checks (if enabled in config)
  #  2) Run model-based guard (Llama-Guard) and return contextual denial if flagged
  #  """
    # 1) Quick regex blocking (optional)
 #   if CFG.get("guard", {}).get("input_regex_block", True):
  #      if _regex_check(prompt):
   #         print("[LlamaGuard] Blocked by regex pattern match.")
    #        return contextual_denial(
     #           reason="input_pattern_match",
      #          alternative="I can provide safe, non-actionable information instead."
       #     )

    # 2) Model-based guard
    #flagged, score, raw = _guard_classify(prompt)
    #if flagged:
     #   print(f"[LlamaGuard] Blocked by model (score={score:.2f}).")
      #  return contextual_denial(
       #     reason="input_guard_model",
        #    alternative="I can provide safe, non-actionable information instead."
        #)

    # Allowed
    #return {"allowed": True}







#"""
#Input guard using meta-llama/Llama-Guard-4-12B.
#Performs regex checks first (cheap), then calls the guard model (bfloat16, device_map="auto").
#Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
#"""

#import os
#import re
#import torch
#import time
#import warnings
#from typing import Tuple
#from src.defenses import contextual_denial
#from src.constant import CFG

# Suppress deprecation warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)

# simple regex patterns (cheap first-line defense)
#SENSITIVE_PATTERNS = [
 #   r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
  #  r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
  #  r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
  #  r"\bwhy are\b.*(bad at|worse at|less intelligent)",
  #  r"\bgive me\b.*(phone numbers|email addresses|home address)",
  #  r"\b(list|give|show)\b.*\b(reasons|ways)\b.*\b(people|those|people from)\b"
#]

# lazy-loaded guard (model, processor tuple)
#_guard_model = None
#_guard_processor = None


#def _load_guard():
 #   """
  #  Load the Llama-Guard-4 model into GPU (bfloat16). Uses device_map='auto'.
  #  Returns (model, processor) tuple or (None, None) on failure.
  #  """
  #  global _guard_model, _guard_processor
  #  if _guard_model is not None and _guard_processor is not None:
   #     return _guard_model, _guard_processor

   # try:
    #    from transformers import AutoProcessor, Llama4ForConditionalGeneration, AutoConfig
        
        # Use Llama-Guard-4-12B
     #   model_name = CFG.get("guard", {}).get("guard_model", "meta-llama/Llama-Guard-4-12B")
        
      #  print(f"[LlamaGuard] Loading {model_name}...")
        
        # Load processor (Llama-Guard-4 uses AutoProcessor, not AutoTokenizer)
       # _guard_processor = AutoProcessor.from_pretrained(model_name)
        
        # Load config and set attention_chunk_size (required for Llama 4)
      #  config = AutoConfig.from_pretrained(model_name)
      #  if not hasattr(config, 'attention_chunk_size') or config.attention_chunk_size is None:
       #     config.attention_chunk_size = 2048
        #    print(f"[LlamaGuard] Set attention_chunk_size=2048 for Llama 4")
        #
        # Load model (use Llama4ForConditionalGeneration for Llama-Guard-4)
       # _guard_model = Llama4ForConditionalGeneration.from_pretrained(
        #    model_name,
         #   config=config,
          #  device_map="auto",
           # torch_dtype=torch.bfloat16,
        #)
        
        # Set to eval mode
        #_guard_model.eval()
        
        #print(f"[LlamaGuard] Loaded {model_name} (bfloat16, device_map=auto)")
        #return _guard_model, _guard_processor
        
    #except Exception as e:
     #   print(f"[LlamaGuard] Failed to load guard model: {e}")
      #  import traceback
       # traceback.print_exc()
        #_guard_model = None
        #_guard_processor = None
        #return None, None


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
    #model, processor = _load_guard()
    #if model is None or processor is None:
     #   return False, 0.0, ""

   # try:
    #    start_time = time.time()
     #   
        # Format messages exactly like HuggingFace example
     #   messages = [
      #      {
       #         "role": "user",
        #        "content": [
         #           {"type": "text", "text": prompt}
          #      ]
           # },
        #]
        
        # Apply chat template using processor (not tokenizer)
        #inputs = processor.apply_chat_template(
         #   messages,
          #  tokenize=True,
           # add_generation_prompt=True,
           # return_tensors="pt",
           # return_dict=True,
        #).to(model.device)
        
        # Generate with cache disabled (fixes StaticCache error)
        #with torch.no_grad():
         #   outputs = model.generate(
          #      **inputs,
           #     max_new_tokens=10,
            #    do_sample=False,
             #   use_cache=False,  # CRITICAL: Disable cache to avoid StaticCache error
            #)
        
        # Decode only the new tokens (exactly like HuggingFace example)
        #response = processor.batch_decode(
         #   outputs[:, inputs["input_ids"].shape[-1]:], 
          #  skip_special_tokens=True
        #)[0]
        
        #print(f"[DEBUG] Raw guard output:\n{response}\n")

        #elapsed = time.time() - start_time
        #if elapsed > 20:
         #   print(f"[LlamaGuard] Warning: classification took {elapsed:.1f}s")

        # Analyze output - Llama Guard returns "safe" or "unsafe\nS#"
        #text = response.lower().strip()
        
        # Direct check for Llama Guard's output format
        #if text.startswith("safe"):
         #   flagged = False
          #  score = 0.0
        #elif text.startswith("unsafe"):
         #   flagged = True
          #  score = 1.0
        #else:
            # Fallback: keyword-based scoring if output format is unexpected
         #   unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]
          #  score = float(sum(1.0 for w in unsafe_words if w in text) / max(1, len(unsafe_words)))
            
            # Get threshold
          #  cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.55)

          #  if threshold is not None:
           #     try:
            #        guard_thresh = float(threshold)
             #   except (TypeError, ValueError):
              #      print("[LlamaGuard] Warning: Invalid threshold provided, using default 0.55")
               #     guard_thresh = 0.55
            #else:
             #   try:
              #      guard_thresh = float(cfg_thresh) if cfg_thresh is not None else 0.55
               # except (TypeError, ValueError):
                #    print("[LlamaGuard] Warning: Invalid config threshold, using default 0.55")
                 #   guard_thresh = 0.55

            #flagged = bool(score >= guard_thresh)

        #print(f"[DEBUG] score={score:.3f}, flagged={flagged}")

        #return flagged, score, response

    #except Exception as e:
     #   print(f"[LlamaGuard] Classification error: {e}")
      #  import traceback
       # traceback.print_exc()
        #return False, 0.0, ""


#def check_input(prompt: str):
 #   """
  #  Public function used by the pipeline.
   # 1) Run fast regex checks (if enabled in config)
   # 2) Run model-based guard (Llama-Guard) and return contextual denial if flagged
   # """
    # 1) Quick regex blocking (optional)
   # if CFG.get("guard", {}).get("input_regex_block", True):
    #    if _regex_check(prompt):
     #       print("[LlamaGuard] Blocked by regex pattern match.")
      #      return contextual_denial(
       #         reason="input_pattern_match",
        #        alternative="I can provide safe, non-actionable information instead."
         #   )

    # 2) Model-based guard
    #flagged, score, raw = _guard_classify(prompt)
    #if flagged:
     #   print(f"[LlamaGuard] Blocked by model (score={score:.2f}).")
      #  return contextual_denial(
       #     reason="input_guard_model",
        #    alternative="I can provide safe, non-actionable information instead."
        #)

    # Allowed
    #return {"allowed": True}









"""
Input guard using meta-llama/Llama-Guard-3-8B (stable and simple).
Performs regex checks first (cheap), then calls the guard model (bfloat16, device_map="auto").
Designed for machines with plenty of GPU memory (RTX 4090 / 5090).
"""

import os
import re
import torch
import time
import warnings
from typing import Tuple
from src.defenses import contextual_denial
from src.constant import CFG

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# simple regex patterns (cheap first-line defense)
SENSITIVE_PATTERNS = [
    r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
    r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
    r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
    r"\bwhy are\b.*(bad at|worse at|less intelligent)",
    r"\bgive me\b.*(phone numbers|email addresses|home address)",
    r"\b(list|give|show)\b.*\b(reasons|ways)\b.*\b(people|those|people from)\b"
]

# lazy-loaded guard
_guard_model = None
_guard_tokenizer = None


def _load_guard():
    """
    Load the Llama-Guard model into GPU (bfloat16). Uses device_map='auto'.
    Returns (model, tokenizer) tuple or (None, None) on failure.
    """
    global _guard_model, _guard_tokenizer
    if _guard_model is not None and _guard_tokenizer is not None:
        return _guard_model, _guard_tokenizer

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use Llama-Guard-3-8B (stable and works without issues)
        model_name = CFG.get("guard", {}).get("guard_model", "meta-llama/Llama-Guard-3-8B")
        
        print(f"[LlamaGuard] Loading {model_name}...")
        
        # Load tokenizer
        _guard_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model - simple and straightforward
        _guard_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # Set to eval mode
        _guard_model.eval()
        
        print(f"[LlamaGuard] Loaded {model_name} (bfloat16, device_map=auto)")
        return _guard_model, _guard_tokenizer
        
    except Exception as e:
        print(f"[LlamaGuard] Failed to load guard model: {e}")
        import traceback
        traceback.print_exc()
        _guard_model = None
        _guard_tokenizer = None
        return None, None


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
    model, tokenizer = _load_guard()
    if model is None or tokenizer is None:
        return False, 0.0, ""

    try:
        start_time = time.time()
        
        # Format messages for Llama Guard
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        
        # Generate - disable cache to avoid errors
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                use_cache=False,  # Fixes cache errors
            )
        
        # Decode only the new tokens
        response = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )[0]
        
        print(f"[DEBUG] Raw guard output:\n{response}\n")

        elapsed = time.time() - start_time
        if elapsed > 20:
            print(f"[LlamaGuard] Warning: classification took {elapsed:.1f}s")

        # Analyze output - Llama Guard returns "safe" or "unsafe\nS#"
        text = response.lower().strip()
        
        # Check for Llama Guard's output format
        if text.startswith("safe"):
            flagged = False
            score = 0.0
        elif text.startswith("unsafe"):
            flagged = True
            score = 1.0
        else:
            # Fallback: keyword-based scoring
            unsafe_words = ["unsafe", "disallowed", "i cannot", "i can't", "refuse", "not allowed", "deny", "cannot comply"]
            score = float(sum(1.0 for w in unsafe_words if w in text) / max(1, len(unsafe_words)))
            
            # Get threshold
            cfg_thresh = CFG.get("guard", {}).get("guard_threshold", 0.4)

            if threshold is not None:
                try:
                    guard_thresh = float(threshold)
                except (TypeError, ValueError):
                    print("[LlamaGuard] Warning: Invalid threshold provided, using default 0.4")
                    guard_thresh = 0.4
            else:
                try:
                    guard_thresh = float(cfg_thresh) if cfg_thresh is not None else 0.4
                except (TypeError, ValueError):
                    print("[LlamaGuard] Warning: Invalid config threshold, using default 0.4")
                    guard_thresh = 0.4

            flagged = bool(score >= guard_thresh)

        print(f"[DEBUG] score={score:.3f}, flagged={flagged}")

        return flagged, score, response

    except Exception as e:
        print(f"[LlamaGuard] Classification error: {e}")
        import traceback
        traceback.print_exc()
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
