# models/llama_guard.py
# Simple rule-based input filter simulating LlamaGuard behavior.
# Replace with real model-based guard later.

import re
from src.defenses import contextual_denial

SENSITIVE_PATTERNS = [
    r"\bhow to\b.*(bomb|weapon|hack|steal|bypass|lockpick)",
    r"\bhow can i\b.*(hack|bypass|steal|evade|get away with)",
    r"\bwhat is\b.*(address|phone|ssn|social security|home address)",
    r"\bwhy are\b.*(bad at|worse at|less intelligent)",
    r"\bgive me\b.*(phone numbers|email addresses|home address)"
]

def check_input(prompt: str):
    low = prompt.lower()
    for pat in SENSITIVE_PATTERNS:
        if re.search(pat, low):
            return contextual_denial(reason="input_pattern_match", alternative="I can provide safe, non-actionable information instead.")
    # If you later integrate LlamaGuard model, call it here and return contextual_denial on unsafe verdicts.
    return {"allowed": True}
