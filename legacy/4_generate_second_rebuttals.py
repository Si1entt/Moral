import json
import os
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = "gpt-4o"
INPUT_FILE = "step2_steered_responses.json"
OUTPUT_FILE = "step4_second_rebuttals.json"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_second_rebuttal(context, category, statement, first_rebuttal, first_response):
    prompt = f"""
    You are an expert, critical debater. 
    Context: "{context}"

    Here is the debate history so far:
    1. The person stated: "{statement}"
    2. You rebutted with: "{first_rebuttal}"
    3. The person defended their stance with: "{first_response}"

    Task: Write a concise, one-sentence counter-rebuttal that directly attacks the logic or moral justification in their latest response ("{first_response}"). Keep the focus on the moral theme of {category}. Be sharp, relentless, and critical.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a sharp and relentless debater."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating second rebuttal: {e}")
        return "Your defense still fails to address the core ethical issue of your actions."


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}. Please run step 2 first.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Generating second round of rebuttals for {len(data)} samples...")
    for entry in tqdm(data):
        category = entry.get('category', 'Unknown')
        context = entry.get('context', '')
        statement = entry.get('original_statement', '')
        first_rebuttal = entry.get('rebuttal', '')
        first_response = entry.get('steered_response_to_rebuttal', '')

        # 生成第二轮反驳
        second_rebuttal = generate_second_rebuttal(context, category, statement, first_rebuttal, first_response)
        entry['second_rebuttal'] = second_rebuttal

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Second rebuttals saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()