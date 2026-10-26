"""
实验名称：跨语言/跨数据集线性探针泛化对比实验
假设：
1. "Virtue" (好) 具有较高的跨语言一致性，迁移损失 (Transfer Drop) 较小。
2. "Vice" (坏) 具有较高的文化特异性和离散性，迁移损失较大。

实验流程：
1. Source Domain: MFD (英文)
2. Target Domain: C-MFD (中文)
3. 方法: 
   - A组 (In-Domain): Train(Zh) -> Test(Zh) [作为基准上限]
   - B组 (Transfer): Train(En) -> Test(Zh) [测试泛化]
   - 计算指标: Relative Performance Drop = (A - B) / A
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 基础工具类 (复用之前的加载逻辑)
# =============================================================================

def load_data_pairs(mfd_path, cmfd_path):
    """加载并标准化数据"""
    # 简化加载逻辑，假设 CSV 格式为 standard
    def _process(path, encodings=['utf-8', 'gbk', 'gb18030']):
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc)
                if len(df.columns) < 2: continue
                df.columns = ['word', 'foundation']
                # 标准化标签
                df['category'] = df['foundation'].apply(lambda x: x.split('.')[0])
                df['polarity'] = df['foundation'].apply(lambda x: x.split('.')[1])
                replacements = {'sanc': 'sanctity', 'auth': 'authority', 'loya': 'loyalty', 'fair': 'fairness'}
                df['category'] = df['category'].replace(replacements)
                return df
            except:
                continue
        raise ValueError(f"无法读取文件: {path}")

    mfd = _process(mfd_path)
    cmfd = _process(cmfd_path)
    return mfd, cmfd

class RepresentationExtractor:
    """提取 Llama-3.1 最后一层表征"""
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_batch(self, words: List[str], lang: str) -> Dict[str, np.ndarray]:
        """批量提取"""
        res = {}
        batch_size = 32 # 可根据显存调整
        
        for i in tqdm(range(0, len(words), batch_size), desc=f"Extracting ({lang})"):
            batch_words = words[i:i+batch_size]
            # Prompt Engineering: 引导模型输出道德相关的语义
            if lang == 'en':
                prompts = [f"Your first instinct for the word  \"{w}\" is: " for w in batch_words]
            else:
                prompts = [f"你对“{w}”这个词的第一直觉是：" for w in batch_words]
                
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # 提取最后一层，最后一个 Token
            last_hidden = outputs.hidden_states[-1] # (Batch, Seq, Dim)
            # 注意: padding位置处理。这里简单取最后一个非pad token，或者直接取 -1 (因为是 left padding 或者 causal mask)
            # Llama 通常没有 pad token 或者使用 EOS。简单起见取 -1
            vecs = last_hidden[:, -1, :].cpu().float().numpy()
            
            for w, v in zip(batch_words, vecs):
                res[w] = v
        return res

# =============================================================================
# 2. 实验核心：跨语言迁移
# =============================================================================

class CrossLingualProbe:
    def __init__(self, mfd_df, cmfd_df, mfd_vecs, cmfd_vecs):
        self.en_df = mfd_df
        self.zh_df = cmfd_df
        self.en_vecs = mfd_vecs
        self.zh_vecs = cmfd_vecs
        self.categories = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
        
    def _get_dataset(self, df, vecs, category, polarity):
        """构建二分类数据集"""
        # 正样本
        pos_mask = (df['category'] == category) & (df['polarity'] == polarity)
        pos_words = df[pos_mask]['word'].tolist()
        X_pos = np.array([vecs[w] for w in pos_words if w in vecs])
        
        if len(X_pos) < 5: return None, None
        
        # 负样本 (随机抽取同语言下的其他词，1:1平衡)
        neg_mask = ~pos_mask
        neg_words = df[neg_mask]['word'].tolist()
        # 确保负样本数量不超过正样本太多，或者刚好平衡
        if len(neg_words) > len(X_pos):
            sampled_neg_words = np.random.choice(neg_words, size=len(X_pos), replace=False)
        else:
            sampled_neg_words = neg_words
            
        X_neg = np.array([vecs[w] for w in sampled_neg_words if w in vecs])
        
        # 再次检查平衡后的数量
        min_len = min(len(X_pos), len(X_neg))
        X_pos = X_pos[:min_len]
        X_neg = X_neg[:min_len]
        
        X = np.concatenate([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))
        
        return X, y

    def _run_single_direction(self, X_source, y_source, X_target, y_target):
        """
        运行单向迁移实验
        返回: (In-Domain Acc, Transfer Acc, Drop)
        """
        # 1. In-Domain Baseline (在 Target 上训练，在 Target 上测)
        # 使用 5-Fold CV 估算 Target 域内的理论上限
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        in_domain_scores = []
        
        # 如果样本太少无法做5折，退化为3折或2折
        n_splits = 10
        # if len(y_target) < 10: n_splits = 2
        # elif len(y_target) < 25: n_splits = 3
        
        try:
            kf_adj = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_idx, test_idx in kf_adj.split(X_target, y_target):
                clf = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1)
                clf.fit(X_target[train_idx], y_target[train_idx])
                in_domain_scores.append(accuracy_score(y_target[test_idx], clf.predict(X_target[test_idx])))
            acc_in_domain = np.mean(in_domain_scores)
        except:
            acc_in_domain = np.nan

        # 2. Transfer (在 Source 上训练，在 Target 上测)
        clf_transfer = LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1)
        clf_transfer.fit(X_source, y_source)
        y_pred = clf_transfer.predict(X_target)
        acc_transfer = accuracy_score(y_target, y_pred)
        
        return acc_in_domain, acc_transfer

    def run_experiment(self):
        results = []
        print("\n=== 开始双向跨语言迁移实验 ===")
        
        for cat in tqdm(self.categories, desc="Processing Categories"):
            for polarity in ['virtue', 'vice']:
                
                # 准备数据
                X_en, y_en = self._get_dataset(self.en_df, self.en_vecs, cat, polarity)
                X_zh, y_zh = self._get_dataset(self.zh_df, self.zh_vecs, cat, polarity)
                
                if X_en is None or X_zh is None:
                    continue
                
                # --- 方向 1: En -> Zh ---
                base_zh, trans_en2zh = self._run_single_direction(X_en, y_en, X_zh, y_zh)
                # if not np.isnan(base_zh):
                #     results.append({
                #         'Category': cat,
                #         'Polarity': polarity,
                #         'Direction': 'En->Zh',
                #         'In_Domain_Acc': base_zh,      # Zh训练Zh测
                #         'Transfer_Acc': trans_en2zh,   # En训练Zh测
                #         'Drop': base_zh - trans_en2zh  # 迁移损失
                #     })
                
                # --- 方向 2: Zh -> En ---
                base_en, trans_zh2en = self._run_single_direction(X_zh, y_zh, X_en, y_en)
                # if not np.isnan(base_en):
                #     results.append({
                #         'Category': cat,
                #         'Polarity': polarity,
                #         'Direction': 'Zh->En',
                #         'In_Domain_Acc': base_en,      # En训练En测
                #         'Transfer_Acc': trans_zh2en,   # Zh训练En测
                #         'Drop': base_en - trans_zh2en  # 迁移损失
                #     })

                if not np.isnan(base_en):
                    results.append({
                        'Category': cat,
                        'Polarity': polarity,
                        'Direction': 'En->Zh',
                        'In_Domain_Acc': base_en,      # En训练En测
                        'Transfer_Acc': trans_en2zh,   # En训练Zh测
                        'Drop': base_en - trans_en2zh  # 迁移损失
                    })

                if not np.isnan(base_zh):
                    results.append({
                        'Category': cat,
                        'Polarity': polarity,
                        'Direction': 'Zh->En',
                        'In_Domain_Acc': base_zh,      # Zh训练Zh测
                        'Transfer_Acc': trans_zh2en,   # Zh训练En测
                        'Drop': base_zh - trans_zh2en  # 迁移损失
                    })
                    
        return pd.DataFrame(results)

# =============================================================================
# 3. 可视化与主程序
# =============================================================================

def visualize_results(df, output_dir):
    if df.empty: return
    
    # 1. 核心验证图：Virtue vs Vice 的迁移损失 (Drop)
    # 我们将两个方向的数据合并，看总体的 Virtue vs Vice 差异
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, 
        x='Polarity', y='Drop', 
        hue='Direction', # 区分 En->Zh 和 Zh->En
        capsize=.1, errorbar="sd", palette="Set2"
    )
    plt.title("双向跨语言迁移性能损失 (Performance Drop)\n(Hypothesis: Vice Drop > Virtue Drop in BOTH directions)")
    plt.ylabel("Accuracy Drop (In-Domain - Transfer)")
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'bidirectional_drop.png'))
    plt.show()

    # 2. 细节图：按 Category 拆分
    # 使用 FacetGrid 展示每个道德基础的详细情况
    g = sns.catplot(
        data=df, kind="bar",
        x="Direction", y="Drop", hue="Polarity",
        col="Category", col_wrap=3, height=4, aspect=0.8,
        palette={'virtue': '#2ecc71', 'vice': '#e74c3c'}, # 绿好红坏
        capsize=.1
    )
    g.fig.suptitle('Performance Drop per Category (Lower is Better Transfer)', y=1.02)
    g.set_axis_labels("", "Accuracy Drop")
    plt.savefig(os.path.join(output_dir, 'bidirectional_category_detail.png'))
    plt.show()

    # 3. 统计摘要
    print("\n=== 结果摘要 (按方向与极性) ===")
    summary = df.groupby(['Direction', 'Polarity'])[['In_Domain_Acc', 'Transfer_Acc', 'Drop']].mean()
    print(summary)
    
    # 保存
    df.to_csv(os.path.join(output_dir, 'bidirectional_results.csv'), index=False)

def main():
    # 配置路径
    MFD_PATH = "datasets/MFD.csv"
    CMFD_PATH = "datasets/C-MFD.csv"
    MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B") 
    OUTPUT_DIR = "results/cross_lingual_experiment"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    mfd_df, cmfd_df = load_data_pairs(MFD_PATH, CMFD_PATH)
    
    # 2. 提取特征 (建议提取一次后保存为 .npy，下次直接加载)
    extractor = RepresentationExtractor(MODEL_PATH)
    
    # 检查是否有缓存
    if os.path.exists(os.path.join(OUTPUT_DIR, "mfd_vecs.npy")):
        print("Loading cached vectors...")
        mfd_vecs = np.load(os.path.join(OUTPUT_DIR, "mfd_vecs.npy"), allow_pickle=True).item()
        cmfd_vecs = np.load(os.path.join(OUTPUT_DIR, "cmfd_vecs.npy"), allow_pickle=True).item()
    else:
        mfd_vecs = extractor.extract_batch(mfd_df['word'].unique().tolist(), 'en')
        cmfd_vecs = extractor.extract_batch(cmfd_df['word'].unique().tolist(), 'zh')
        # 缓存
        np.save(os.path.join(OUTPUT_DIR, "mfd_vecs.npy"), mfd_vecs)
        np.save(os.path.join(OUTPUT_DIR, "cmfd_vecs.npy"), cmfd_vecs)
    
    # 3. 运行实验
    exp = CrossLingualProbe(mfd_df, cmfd_df, mfd_vecs, cmfd_vecs)
    results_df = exp.run_experiment()
    
    # 4. 可视化
    visualize_results(results_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()