# run_pipeline.py
# Top-level script to run the full pipeline: red-team + evaluate
#from src.red_team import run_once
#from src.evaluate import compute_metrics

#def main():
 #   print("Starting Ethical AI Red Teaming pipeline...")
  #  run_once()
   # print("\nEvaluating results...")
   # compute_metrics()
   # print("\nDone. See data/ folder for outputs.")

#if __name__ == "__main__":
 #   main()



# run_pipeline.py
#from src.red_team import run_once
#from src.evaluate import compute_metrics
#from src.utils import ensure_dirs

#def main():
 #   print("Starting Ethical AI Red Teaming pipeline...")
  #  ensure_dirs()
  #  run_once()
  #  print("\nEvaluating results...")
  #  compute_metrics()
  #  print("\nDone. Check data/ for outputs.")

#if __name__ == "__main__":
 #   main()




# run_pipeline.py
"""
Runs the full Ethical AI Red Teaming pipeline:
1. Load prompts
2. Run input filter (LlamaGuard)
3. Generate model responses
4. Run output scanning (Detoxify + Toxic-BERT)
5. Log results and generate compliance report
"""

import os
from src.red_team import run_once
from src.evaluate import compute_metrics
from src.utils import ensure_dirs

def main():
    print("🚀 Starting Ethical AI Red Teaming pipeline...\n")
    ensure_dirs()
    
    # Step 1: Run Red Team evaluation
    print("🧠 Running red team tests...")
    run_once()
    
    # Step 2: Evaluate performance
    print("\n📊 Generating compliance report...")
    compute_metrics()
    
    print("\n✅ Pipeline complete! Check your reports in data/reports/")

if __name__ == "__main__":
    main()


