# models/output_scanner.py
from detoxify import Detoxify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# load once (on init)
detox = Detoxify('original')  # small and fast
HF_MODEL = "unitary/toxic-bert"  # HF toxic model
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).to(device).eval()

#hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).eval().to("cuda")

# thresholds (tune these in config.yaml)
DETOX_THRESH = 0.5    # Detoxify label threshold for any toxic label
HF_TOXICITY_THRESH = 0.55  # toxic-bert threshold on toxic score

def scan_with_detox(text: str):
    """
    Returns dict of detox scores, and bool flagged if any score > DETOX_THRESH.
    """
    scores = detox.predict(text)  # returns dict: toxic, severe_toxic, obscene, insult, threat, identity_hate
    flagged = any(v >= DETOX_THRESH for v in scores.values())
    return {"scores": scores, "flagged": bool(flagged)}

def scan_with_hf_toxicbert(text: str):
    """
    Runs a HF toxic classifier and returns toxic probability.
    Assumes binary classification labels [non-toxic, toxic] or similar.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        logits = hf_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    # many HF toxic models use label 1 as toxic; check the model card if different
    toxic_prob = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])
    flagged = toxic_prob >= HF_TOXICITY_THRESH
    return {"toxic_prob": toxic_prob, "flagged": bool(flagged)}

def output_scan(text: str):
    """
    Combined scanner: runs Detoxify and Toxic-BERT and returns decision.
    """
    detox_res = scan_with_detox(text)
    hf_res = scan_with_hf_toxicbert(text)
    # combine logic: flag if either tool flags
    flagged = detox_res["flagged"] or hf_res["flagged"]
    meta = {"detox": detox_res, "hf": hf_res}
    return {"flagged": flagged, "meta": meta}





#This we have made commented 

# models/output_scanner.py
# Uses Detoxify + a HF toxic model to flag toxic outputs.
# This file handles GPU usage if available.

#from detoxify import Detoxify
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#import torch
#import os
#from src.constants import CFG
#
## load detoxify (uses CPU by default; model is small)
#try:
#    detox = Detoxify('original')
#except Exception as e:
#    print("Warning: Detoxify failed to initialize:", e)
#    detox = None
#
#HF_MODEL_NAME = "unitary/toxic-bert"  # can be changed in config later
#device = "cuda" if torch.cuda.is_available() else "cpu"
#
#try:
#    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
#    hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME).to(device).eval()
#except Exception as e:
#    print("Warning: Could not load HF toxic model:", e)
#    tokenizer = None
#    hf_model = None
#
#DETOX_THRESH = CFG["safety"]["detox_threshold"]
#HF_TOX_THRESH = CFG["safety"]["hf_toxic_threshold"]
#
#def scan_with_detox(text: str):
#    if detox is None:
#        return {"flagged": False, "scores": {}}
#    scores = detox.predict(text)
#    flagged = any(v >= DETOX_THRESH for v in scores.values())
#    return {"flagged": bool(flagged), "scores": scores}
#
#def scan_with_hf(text: str):
#    if hf_model is None or tokenizer is None:
#        return {"flagged": False, "toxic_prob": 0.0}
#    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
#    with torch.no_grad():
#        logits = hf_model(**inputs).logits
#        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#    # Many toxic models are binary: [non-toxic, toxic]; if not, adapt accordingly
#    toxic_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
#    flagged = toxic_prob >= HF_TOX_THRESH
#    return {"flagged": bool(flagged), "toxic_prob": toxic_prob}
#
#def output_scan(text: str):
#    detox_res = scan_with_detox(text)
#    hf_res = scan_with_hf(text)
#    flagged = detox_res["flagged"] or hf_res["flagged"]
#    meta = {"detox": detox_res, "hf": hf_res}
#    return {"flagged": bool(flagged), "meta": meta}
