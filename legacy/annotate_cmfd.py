
import os
import pandas as pd
import time
from openai import OpenAI

# ================= 配置区域 =================
# 请在此处填入您的大模型 API 信息
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")  # 如果使用其他服务商（如DeepSeek, Moonshot等），请修改此URL
MODEL_NAME = "gpt-4o"  # 或者 gpt-3.5-turbo, deepseek-chat 等

# 输入和输出文件路径
INPUT_FILE = "./datasets/C-MFD.csv"
OUTPUT_FILE = "C-MFD_labeled.csv"

CATEGORY_MAP = {
    'care': 'Care (关爱)',
    'fair': 'Fairness (公平)',
    'loya': 'Loyalty (忠诚)',
    'auth': 'Authority (权威)',
    'sanc': 'Sanctity (圣洁)',
    'libe': 'Liberty (自由)',
    'altr': 'Altruism (利他)',
    'dili': 'Diligence (勤勉)',
    'resi': 'Resilience (坚毅)',
    'mode': 'Modesty (谦逊)',
    'wast': 'Waste (挥霍)',
    'general': 'General Morality (通用道德)'
}

# ===========================================

def get_polarity(word, foundation_abbr, client):
    """
    调用大模型判断词汇是 Virtue (正面) 还是 Vice (负面)
    """

    foundation_full = CATEGORY_MAP.get(foundation_abbr, foundation_abbr)

    prompt = f"""
    任务：判断中文词汇 "{word}" 在道德基础类别 "{foundation_full}" 中，属于正面（virtue/美德/积极）还是负面（vice/恶行/消极）。

    请严格遵守以下输出格式：
    - 如果是正面，仅输出: virtue
    - 如果是负面，仅输出: vice
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个道德基础字典（MFD）的标注专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # 降低随机性
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().lower()
        # print(result)

        # 简单的后处理，防止模型输出标点符号
        if "virtue" in result:
            return "virtue"
        elif "vice" in result:
            return "vice"
        else:
            return "unknown"

    except Exception as e:
        print(f"Error processing {word}: {e}")
        return "error"


def main():
    # 1. 读取数据
    print(f"正在读取 {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 检查列名，确保包含 'chinese' 和 'foundation'
    if 'chinese' not in df.columns or 'foundation' not in df.columns:
        print("错误：CSV文件需要包含 'chinese' 和 'foundation' 列")
        return

    # 初始化客户端
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"开始处理 {len(df)} 条数据... (建议先用 head() 测试少量数据)")

    # 为了演示和测试，您可以先取消下面这行的注释来只跑前 10 条
    # df = df.head(10)

    new_foundations = []

    # 2. 循环处理
    # 使用 tqdm 显示进度条（如果已安装），否则使用普通打印
    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df))
    except ImportError:
        iterator = df.iterrows()
        print("提示：安装 tqdm 库可以显示进度条 (pip install tqdm)")

    for index, row in iterator:
        word = row['chinese']
        base_foundation = row['foundation']

        # 调用 API
        polarity = get_polarity(word, base_foundation, client)

        # 拼接结果，例如 care.virtue
        if polarity in ['virtue', 'vice']:
            new_val = f"{base_foundation}.{polarity}"
        else:
            # 如果API报错或无法判断，保留原样或标记为 unknown
            new_val = f"{base_foundation}.{polarity}"

        new_foundations.append(new_val)

        # 避免触发 API 速率限制，可以适当休眠
        # time.sleep(0.1)

    # 3. 保存结果
    df['foundation_labeled'] = new_foundations

    # 如果您想直接覆盖 foundation 列，可以使用下面这行：
    # df['foundation'] = new_foundations

    print(f"正在保存结果到 {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print("完成！")


if __name__ == "__main__":
    # 检查是否安装了 openai 库
    try:
        import openai

        main()
    except ImportError:
        print("请先安装 openai 库: pip install openai pandas")