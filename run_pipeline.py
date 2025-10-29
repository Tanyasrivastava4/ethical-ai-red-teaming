# run_pipeline.py
# Top-level script to run the full pipeline: red-team + evaluate
from src.red_team import run_once
from src.evaluate import compute_metrics

def main():
    print("Starting Ethical AI Red Teaming pipeline...")
    run_once()
    print("\nEvaluating results...")
    compute_metrics()
    print("\nDone. See data/ folder for outputs.")

if __name__ == "__main__":
    main()


