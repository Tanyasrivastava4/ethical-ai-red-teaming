# src/constants.py
import yaml, os
cfg_path = "configs/config.yaml"
if not os.path.exists(cfg_path):
    raise FileNotFoundError("Please create configs/config.yaml")

with open(cfg_path, "r") as f:
    CFG = yaml.safe_load(f)
