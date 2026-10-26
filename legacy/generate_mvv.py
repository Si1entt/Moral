import pandas as pd
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm  # 用于显示进度条

# ================= 配置区域 =================
# 在此处填入你的 API Key 和 Base URL
API_KEY = os.environ.get("OPENAI_API_KEY", "")
# API_KEY = "<removed>"
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 如果是用其他模型（如DeepSeek/Claude中转），请修改此处
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "gpt-4o"  # 或 "gpt-3.5-turbo", "deepseek-chat" 等
# MODEL_NAME = "qwen3-max"

INPUT_CSV = "./datasets/C-MFD.csv"
PROMPT_TEMPLATE_FILE = "prompts/vector_generation_prompt_chinese.txt"
OUTPUT_FILE = "cmfd_generated_results.jsonl"  # 结果将保存为 JSON Lines 格式
# ===========================================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def expand_category_name(foundation_code):
    """
    将 MFD 的缩写转换为全称
    例如: 'loya.vice' -> 'loyalty.vice'
    """
    mapping = {
        'care': 'care',
        'fair': 'fairness',  # 补全 fair -> fairness
        'loya': 'loyalty',  # 补全 loya -> loyalty
        'auth': 'authority',  # 补全 auth -> authority
        'sanc': 'sanctity'  # 补全 sanc -> sanctity
    }

    try:
        # 分割 foundation (例如 "loya" 和 "vice")
        base, polarity = foundation_code.split('.')
        # 查找映射，如果找不到则保持原样
        full_base = mapping.get(base, base)
        return f"{full_base}.{polarity}"
    except ValueError:
        return foundation_code


def generate_data():
    # 1. 读取并处理 CSV 数据
    print("正在读取并处理 CSV 文件...")
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')

    # === 关键步骤：应用类别名称补全 ===
    # 将 foundation 列应用 expand_category_name 函数
    df['full_category'] = df['foundation_labeled'].apply(expand_category_name)

    print(f"共加载 {len(df)} 个词条。")
    print("类别映射示例 (前5行):")
    print(df[['foundation_labeled', 'full_category']].head().to_string())
    print("-" * 30)

    # 2. 读取 Prompt 模板
    with open(PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # 3. 检查断点续传（防止重复扣费）
    processed_pairs = set()
    if os.path.exists(OUTPUT_FILE):
        print("正在检查已生成的数据...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 只有当 output 中同时包含 word 和 category 时才记录
                    if 'target_word' in data and 'category' in data:
                        # 创建唯一标识：(单词, 类别)
                        unique_key = (data['target_word'], data['category'])
                        processed_pairs.add(unique_key)
                except:
                    pass
        print(f"已跳过 {len(processed_pairs)} 条已处理的历史数据。")

    # 4. 开始循环调用 API
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        # 使用 tqdm 显示进度条
        for index, row in tqdm(df.iterrows(), total=len(df)):
            current_word = row['chinese']
            current_category = row['full_category']

            # 如果已经生成过，跳过
            if (current_word, current_category) in processed_pairs:
                continue

            # === 关键步骤：动态构建 Prompt ===
            # 将模板中的 {word} 和 {category} 替换为当前行的真实数据
            final_prompt = prompt_template.replace("{word}", current_word).replace("{category}", current_category)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in dataset generation."},
                        {"role": "user", "content": final_prompt}
                    ],
                    response_format={"type": "json_object"},  # 强制 JSON 格式确保稳定
                    temperature=0.7
                )

                result_content = response.choices[0].message.content

                # 校验 JSON 是否合法
                json_data = json.loads(result_content)

                # 写入文件
                f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                f_out.flush()  # 立即保存

            except Exception as e:
                print(f"\n处理词汇 '{current_word}' 时出错: {e}")
                # 可以在这里加个 time.sleep 稍作等待

    print(f"\n生成完成！结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_data()