# src/defenses.py
import time, os, csv
from pathlib import Path
DENIAL_MSG = "I cannot answer that. I can explain why it's harmful or offer safe alternatives."

DENIAL_LOG = "data/denials_log.csv"

def contextual_denial(reason: str, alternative: str = None):
    msg = DENIAL_MSG
    if alternative:
        msg = f"{msg} {alternative}"
    entry = {"timestamp": time.time(), "reason": reason, "message": msg}
    _log_denial(entry)
    return {"allowed": False, "message": msg, "reason": reason}

def _log_denial(entry):
    os.makedirs(os.path.dirname(DENIAL_LOG), exist_ok=True)
    header = ["timestamp", "reason", "message"]
    write_header = not os.path.exists(DENIAL_LOG) or os.path.getsize(DENIAL_LOG) == 0
    with open(DENIAL_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([entry["timestamp"], entry["reason"], entry["message"]])
