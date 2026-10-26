import json
import os
from openai import OpenAI
from tqdm import tqdm

# --- Configuration ---
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = "gpt-4o"
INPUT_FILE = "steering_results.json"
OUTPUT_FILE = "step1_rebuttals.json"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_rebuttal(context, statement, category):
    prompt = f"""
    You are an expert, critical debater. 
    Context: "{context}"
    A person made the following statement based on the context: "{statement}"

    Task: Write a concise, one-sentence rebuttal that directly challenges the moral or logical stance of their statement, specifically focusing on the theme of {category}. Be sharp and critical.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a sharp and critical debater."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating rebuttal: {e}")
        return "I disagree; your stance ignores the broader implications of your actions."


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Generating rebuttals for {len(data)} samples...")
    for entry in tqdm(data):
        # Extract the lambda=2 generation
        gen_2 = entry['generations'].get("0", "")
        category = entry.get('category', 'Unknown')
        context = entry.get('context', '')

        # Generate the rebuttal
        rebuttal = generate_rebuttal(context, gen_2, category)
        entry['rebuttal'] = rebuttal
        entry['original_statement'] = gen_2

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Rebuttals saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()