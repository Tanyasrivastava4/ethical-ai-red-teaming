# src/utils.py
import csv, os, time
from pathlib import Path

def ensure_dirs():
    Path("data/reports").mkdir(parents=True, exist_ok=True)

def write_csv_header(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
