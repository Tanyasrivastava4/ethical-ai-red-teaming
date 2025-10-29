# src/red_team.py
import csv, os
from models.llama_guard import check_input
from models.model_loader import generate
from models.output_scanner import output_scan
from src.defenses import contextual_denial

PROMPTS_CSV = "data/prompts.csv"
RESULTS_RAW = "data/results_raw.csv"
RESULTS_DEFENDED = "data/results_defended.csv"

os.makedirs("data", exist_ok=True)

def read_prompts(path=PROMPTS_CSV):
    import pandas as pd
    return pd.read_csv(path).to_dict(orient="records")

def run_once():
    prompts = read_prompts()
    # prepare CSVs
    with open(RESULTS_RAW, "w", newline="", encoding="utf-8") as raw_f, \
         open(RESULTS_DEFENDED, "w", newline="", encoding="utf-8") as def_f:
        raw_writer = csv.writer(raw_f)
        def_writer = csv.writer(def_f)
        raw_writer.writerow(["id","prompt","model_reply"])
        def_writer.writerow(["id","prompt","final_reply","denied","deny_reason"])
        for row in prompts:
            pid = row.get("id")
            prompt = row.get("prompt")
            # input check
            input_check = check_input(prompt)
            if not input_check.get("allowed", True):
                # denied before model
                final = input_check["message"]
                def_writer.writerow([pid, prompt, final, True, input_check["reason"]])
                continue
            # model reply
            reply = generate(prompt)
            raw_writer.writerow([pid, prompt, reply])
            # output scan
            scan = output_scan(reply)
            if scan["flagged"]:
                deny = contextual_denial(reason="output_toxic")
                def_writer.writerow([pid, prompt, deny["message"], True, "output_toxic"])
            else:
                def_writer.writerow([pid, prompt, reply, False, ""])
    print("Run complete. Results in data/")

if __name__ == "__main__":
    run_once()
