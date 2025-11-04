# src/constants.py
import yaml, os
cfg_path = "configs/config.yml"
if not os.path.exists(cfg_path):
    raise FileNotFoundError("Please create configs/config.yml")

with open(cfg_path, "r") as f:
    CFG = yml.safe_load(f)
