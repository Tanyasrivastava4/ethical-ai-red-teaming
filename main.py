# main.py
import csv
import json
import os
import sys
from models.llama_guard import check_input
from models.model_loader import load_model, generate_model_response

# -------------------------------
# Helper: run a single prompt
# -------------------------------
def run_prompt(prompt, category, generator):
    print(f"\n🧠 CATEGORY: {category}")
    print(f"🧠 USER PROMPT: {prompt}")

    # Step 1 — Input filtering (LlamaGuard)
    result = check_input(prompt)
    if not result.get("allowed", True):
        print(f"🚫 BLOCKED by LlamaGuard: {result.get('message')}")
        return {
            "category": category,
            "prompt": prompt,
            "allowed": False,
            "reason": result.get("reason"),
            "response": None,
        }

    # Step 2 — Generate model response
    response = generate_model_response(prompt, generator)
    print(f"✅ MODEL RESPONSE:\n{response}")
    return {
        "category": category,
        "prompt": prompt,
        "allowed": True,
        "reason": None,
        "response": response,
    }


# -------------------------------
# Main: read prompts CSV, run, save results
# -------------------------------
if __name__ == "__main__":
    csv_path = "data/prompts.csv"
    results_dir = "results"
    output_csv = os.path.join(results_dir, "output_results.csv")
    output_json = os.path.join(results_dir, "output_results.json")
    results = []

    # Ensure results folder exists
    os.makedirs(results_dir, exist_ok=True)

    # Load CSV and process rows
    try:
        # Open CSV (default comma delimiter). If your file uses a different delimiter,
        # change/remove the `delimiter` parameter.
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)  # default delimiter=','
            # Clean header names (remove leading/trailing spaces)
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]

            # Ensure required columns exist
            required_cols = {"prompt", "category"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                print("❌ CSV is missing required columns. Found columns:", reader.fieldnames)
                print("Required columns: 'prompt' and 'category'")
                sys.exit(1)

            # Load model once (outside loop) to avoid repeated heavy loads
            generator = load_model()

            for row in reader:
                # Safely access and strip values
                prompt = (row.get("prompt") or "").strip()
                category = (row.get("category") or "").strip()
                if not prompt:
                    # skip empty prompts
                    continue

                res = run_prompt(prompt, category, generator)
                results.append(res)

    except FileNotFoundError:
        print(f"❌ Could not find {csv_path}. Make sure it exists and path is correct.")
        sys.exit(1)
    except Exception as e:
        print("❌ Unexpected error while reading CSV or processing prompts:", str(e))
        sys.exit(1)

    # -------------------------------
    # Summary
    # -------------------------------
    allowed = sum(1 for r in results if r.get("allowed"))
    blocked = sum(1 for r in results if not r.get("allowed"))
    total = len(results)
    print("\n📊 Summary:")
    print(f"Total prompts processed: {total}")
    print(f"✅ Allowed: {allowed} | 🚫 Blocked: {blocked}")

    # -------------------------------
    # Save results to CSV
    # -------------------------------
    # Use a fixed set of columns (so CSV writer columns are stable)
    fieldnames = ["category", "prompt", "allowed", "reason", "response"]
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                # Ensure values are serializable / strings
                row = {
                    "category": r.get("category"),
                    "prompt": r.get("prompt"),
                    "allowed": r.get("allowed"),
                    "reason": r.get("reason") or "",
                    "response": (r.get("response") or "").replace("\n", " ").strip(),
                }
                writer.writerow(row)
        print(f"📁 Results saved to {output_csv}")
    except Exception as e:
        print("❌ Failed to write CSV results:", str(e))

    # -------------------------------
    # Save results to JSON
    # -------------------------------
    try:
        with open(output_json, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2, ensure_ascii=False)
        print(f"📁 Results saved to {output_json}")
    except Exception as e:
        print("❌ Failed to write JSON results:", str(e))













# main.py
#import csv
#from models.llama_guard import check_input
#from models.model_loader import load_model, generate_model_response


#def run_prompt(prompt, category):
 #   print(f"\n🧠 CATEGORY: {category}")
  #  print(f"🧠 USER PROMPT: {prompt}")

    # Step 1 — Check input using llama_guard
   # result = check_input(prompt)
   # if not result["allowed"]:
    #    print(f"🚫 BLOCKED by LlamaGuard: {result['message']}")
     #   return {"category": category, "prompt": prompt, "allowed": False, "reason": result["reason"]}

    # Step 2 — Generate model response safely
    #generator = load_model()
    #response = generate_model_response(prompt, generator)
    #print(f"✅ MODEL RESPONSE:\n{response}")
    #return {"category": category, "prompt": prompt, "allowed": True, "response": response}


#if __name__ == "__main__":
 #   csv_path = "data/prompts.csv"
  #  results = []

   # try:
    #    with open(csv_path, newline='', encoding='utf-8') as f:
     #       reader = csv.DictReader(f)
      #      for row in reader:
       #         prompt = row["prompt"]
        #        category = row["category"]
         #       res = run_prompt(prompt, category)
          #      results.append(res)

        # Print summary
       # print("\n📊 Summary:")
       # allowed = sum(1 for r in results if r["allowed"])
       # blocked = sum(1 for r in results if not r["allowed"])
       # print(f"✅ Allowed: {allowed} | 🚫 Blocked: {blocked}")

    #except FileNotFoundError:
     #   print(f"❌ Could not find {csv_path}. Make sure it exists.")







