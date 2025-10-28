# main.py
import csv
from models.llama_guard import check_input
from models.model_loader import load_model, generate_model_response


def run_prompt(prompt, category):
    print(f"\n🧠 CATEGORY: {category}")
    print(f"🧠 USER PROMPT: {prompt}")

    # Step 1 — Check input using llama_guard
    result = check_input(prompt)
    if not result["allowed"]:
        print(f"🚫 BLOCKED by LlamaGuard: {result['message']}")
        return {"category": category, "prompt": prompt, "allowed": False, "reason": result["reason"]}

    # Step 2 — Generate model response safely
    generator = load_model()
    response = generate_model_response(prompt, generator)
    print(f"✅ MODEL RESPONSE:\n{response}")
    return {"category": category, "prompt": prompt, "allowed": True, "response": response}


if __name__ == "__main__":
    csv_path = "data/prompts.csv"
    results = []

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"]
                category = row["category"]
                res = run_prompt(prompt, category)
                results.append(res)

        # Print summary
        print("\n📊 Summary:")
        allowed = sum(1 for r in results if r["allowed"])
        blocked = sum(1 for r in results if not r["allowed"])
        print(f"✅ Allowed: {allowed} | 🚫 Blocked: {blocked}")

    except FileNotFoundError:
        print(f"❌ Could not find {csv_path}. Make sure it exists.")
