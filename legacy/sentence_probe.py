import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 数据加载模块
# =============================================================================

def parse_foundation(category_str: str) -> str:
    """
    从 category 字符串中提取基础类别 (foundation)。
    例如: "care.virtue" -> "care", "fairness.vice" -> "fairness"
    同时处理可能的缩写映射。
    """
    # 移除 .virtue 或 .vice 后缀
    base = category_str.split('.')[0]
    
    # 映射表 (处理可能出现的缩写)
    mapping = {
        'sanc': 'sanctity',
        'auth': 'authority',
        'loya': 'loyalty',
        'fair': 'fairness',
        'care': 'care'
    }
    
    # 如果 base 在映射表中，返回完整形式；否则假设它已经是完整形式
    return mapping.get(base, base)

def load_jsonl_data(file_path: str, dataset_name: str) -> pd.DataFrame:
    """
    加载 JSONL 数据集，并根据后缀类型生成标签。
    规则：
    - context + virtue_suffix -> {foundation}.virtue
    - context + vice_suffix   -> {foundation}.vice
    """
    data = []
    print(f"[{dataset_name}] 正在加载: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return pd.DataFrame()

    valid_foundations = {'care', 'fairness', 'loyalty', 'authority', 'sanctity'}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                if not line.strip(): continue
                item = json.loads(line)
                
                raw_category = item.get('category', '')
                foundation = parse_foundation(raw_category)
                
                # 过滤掉非标准道德基础的数据
                if foundation not in valid_foundations:
                    continue
                
                pairs = item.get('pairs', [])
                
                for pair in pairs:
                    context = pair.get('context', '').strip()
                    virtue_suffix = pair.get('virtue_suffix', '').strip()
                    vice_suffix = pair.get('vice_suffix', '').strip()
                    
                    # 生成 Virtue 样本
                    if virtue_suffix:
                        data.append({
                            'text': f"{context} {virtue_suffix}",
                            'label': f"{foundation}.virtue",
                            'dataset': dataset_name,
                            'original_word': item.get('target_word'),
                            'type': 'virtue'
                        })
                    
                    # 生成 Vice 样本
                    if vice_suffix:
                        data.append({
                            'text': f"{context} {vice_suffix}",
                            'label': f"{foundation}.vice",
                            'dataset': dataset_name,
                            'original_word': item.get('target_word'),
                            'type': 'vice'
                        })
                    break
                        
            except json.JSONDecodeError:
                print(f"跳过错误行: {line_num}")
                continue

    df = pd.DataFrame(data)
    print(f"[{dataset_name}] 加载完成，共 {len(df)} 条样本。")
    if not df.empty:
        print(f"类别分布预览:\n{df['label'].value_counts().head()}")
    return df

# =============================================================================
# 2. 表征提取模块
# =============================================================================

class RepresentationExtractor:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device} | 模型: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_batch_representations(self, sentences: List[str], batch_size: int = 8) -> List[Optional[np.ndarray]]:
        """
        批量提取最后一层 Token 的隐藏状态
        返回: List[Array(num_layers, hidden_dim)]
        """
        all_results = [None] * len(sentences)
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="提取特征"):
            batch_texts = sentences[i:i+batch_size]
            
            try:
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                # outputs.hidden_states 是一个 tuple，包含 (embedding_layer, layer_1, ..., layer_N)
                # 我们通常跳过 embedding 层 (index 0)，只取 Transformer 层
                hidden_states = outputs.hidden_states[1:] 
                
                # 找到每个句子最后一个非 pad token 的位置
                # attention_mask: (batch, seq) -> sum(1) - 1 得到 index
                last_token_indices = inputs['attention_mask'].sum(1) - 1
                
                # Stack layers: (num_layers, batch, hidden)
                # 将 tuple 转为 tensor
                stacked_layers = torch.stack([h for h in hidden_states])
                
                # 提取每个样本的特征
                for batch_idx in range(len(batch_texts)):
                    idx = last_token_indices[batch_idx]
                    # (num_layers, hidden)
                    vec = stacked_layers[:, batch_idx, idx, :].cpu().float().numpy()
                    all_results[i + batch_idx] = vec
                    
            except Exception as e:
                print(f"Batch {i} error: {e}")
                continue
                
        return all_results

# =============================================================================
# 3. 探针模型 & 训练器
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

class ProbeTrainer:
    def __init__(self, input_dim: int, num_classes: int, device: str):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        
    def train_probe(self, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, batch_size=64):
        probe = LinearProbe(self.input_dim, self.num_classes).to(self.device)
        
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)
        # 验证集如果不为空
        if len(X_val) > 0:
            X_v = torch.FloatTensor(X_val).to(self.device)
            y_v = torch.LongTensor(y_val).to(self.device)
        
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_acc = 0.0
        
        for _ in range(epochs):
            probe.train()
            for bx, by in loader:
                optimizer.zero_grad()
                out = probe(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            
            # Validation
            if len(X_val) > 0:
                probe.eval()
                with torch.no_grad():
                    out_v = probe(X_v)
                    preds = out_v.argmax(dim=1)
                    acc = (preds == y_v).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
            else:
                best_acc = 0.0 # No validation set provided
                
        return probe, best_acc

    def predict(self, probe: nn.Module, X: np.ndarray) -> np.ndarray:
        probe.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = probe(X_t)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()

# =============================================================================
# 4. 双语实验主类
# =============================================================================

class DualLangProbeExperiment:
    def __init__(self, en_path: str, zh_path: str, model_path: str, output_dir: str):
        self.en_path = en_path
        self.zh_path = zh_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 10个类别
        self.expected_classes = [
            'care.virtue', 'care.vice', 
            'fairness.virtue', 'fairness.vice',
            'loyalty.virtue', 'loyalty.vice', 
            'authority.virtue', 'authority.vice',
            'sanctity.virtue', 'sanctity.vice'
        ]
        self.le = LabelEncoder()
        self.le.fit(self.expected_classes)
        
        self.results = []

    def process_dataset(self, extractor, df: pd.DataFrame, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """提取特征并对齐标签"""
        if df.empty: return None, None
        
        # 过滤掉不在预期类别中的数据
        df = df[df['label'].isin(self.expected_classes)].reset_index(drop=True)
        if df.empty: return None, None
        
        print(f"\n正在提取 {dataset_name} 特征...")
        features = extractor.extract_batch_representations(df['text'].tolist())
        
        # 清洗无效数据 (None)
        valid_mask = [f is not None for f in features]
        if not any(valid_mask): return None, None
        
        X_list = [features[i] for i, v in enumerate(valid_mask) if v]
        y_list = self.le.transform(df.loc[valid_mask, 'label'])
        
        # Stack: (N_samples, N_layers, Hidden_dim)
        X_stack = np.stack(X_list)
        return X_stack, y_list

    def run(self):
        # 1. 加载数据
        df_en = load_jsonl_data(self.en_path, "English(MFD)")
        df_zh = load_jsonl_data(self.zh_path, "Chinese(CMFD)")
        
        # 2. 初始化模型提取器
        extractor = RepresentationExtractor(self.model_path, self.device)
        
        # 3. 提取特征
        X_en, y_en = self.process_dataset(extractor, df_en, "English")
        X_zh, y_zh = self.process_dataset(extractor, df_zh, "Chinese")
        
        # 释放模型显存以供探针训练
        del extractor.model
        torch.cuda.empty_cache()
        
        # 获取维度信息
        if X_en is not None:
            num_layers = X_en.shape[1]
            hidden_dim = X_en.shape[2]
        elif X_zh is not None:
            num_layers = X_zh.shape[1]
            hidden_dim = X_zh.shape[2]
        else:
            print("错误：没有有效数据！")
            return

        print(f"\n=== 开始层级探针训练 (层数: {num_layers}, 维度: {hidden_dim}) ===")
        print(f"任务: 10分类 (5道德基础 x 2极性)")
        
        # 4. 逐层训练
        # 5-Fold Cross Validation
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        trainer = ProbeTrainer(hidden_dim, len(self.expected_classes), self.device)
        
        for layer in tqdm(range(num_layers), desc="Layer Probe"):
            layer_res = {'layer': layer}
            
            # --- 训练英文探针 ---
            if X_en is not None:
                accs = []
                X_layer = X_en[:, layer, :]
                for train_idx, test_idx in skf.split(X_layer, y_en):
                    probe, acc = trainer.train_probe(
                        X_layer[train_idx], y_en[train_idx],
                        X_layer[test_idx], y_en[test_idx],
                        epochs=30  # 可根据需要调整
                    )
                    accs.append(acc)
                layer_res['acc_en'] = np.mean(accs)
                layer_res['std_en'] = np.std(accs)
            
            # --- 训练中文探针 ---
            if X_zh is not None:
                accs = []
                X_layer = X_zh[:, layer, :]
                for train_idx, test_idx in skf.split(X_layer, y_zh):
                    probe, acc = trainer.train_probe(
                        X_layer[train_idx], y_zh[train_idx],
                        X_layer[test_idx], y_zh[test_idx],
                        epochs=30
                    )
                    accs.append(acc)
                layer_res['acc_zh'] = np.mean(accs)
                layer_res['std_zh'] = np.std(accs)
                
            self.results.append(layer_res)
            
        # 保存结果
        os.makedirs(self.output_dir, exist_ok=True)
        res_df = pd.DataFrame(self.results)
        res_df.to_csv(os.path.join(self.output_dir, 'dual_lang_sentence_probe_results.csv'), index=False)
        print(f"数据已保存至 {self.output_dir}")
        
        self.visualize(res_df)

    def visualize(self, df):
        plt.figure(figsize=(12, 6))
        
        # 绘制英文曲线
        if 'acc_en' in df.columns:
            plt.plot(df['layer'], df['acc_en'], marker='o', label='English (MFD Generated)', linewidth=2, color='#1f77b4')
            if 'std_en' in df.columns:
                plt.fill_between(df['layer'], df['acc_en']-df['std_en'], df['acc_en']+df['std_en'], alpha=0.2, color='#1f77b4')
        
        # 绘制中文曲线
        if 'acc_zh' in df.columns:
            plt.plot(df['layer'], df['acc_zh'], marker='s', label='Chinese (C-MFD Generated)', linewidth=2, color='#ff7f0e')
            if 'std_zh' in df.columns:
                plt.fill_between(df['layer'], df['acc_zh']-df['std_zh'], df['acc_zh']+df['std_zh'], alpha=0.2, color='#ff7f0e')
            
        plt.axhline(y=0.1, color='r', linestyle='--', label='Random Guess (10%)')
        plt.xlabel('Layer Index')
        plt.ylabel('Probe Accuracy (10-Class)')
        plt.title('Layer-wise Sentence Probing: Moral Foundations (10 Classes)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dual_lang_comparison.png'), dpi=150)
        # plt.show() # 如果在服务器运行，请注释掉此行

# =============================================================================
# 5. 运行入口
# =============================================================================

if __name__ == "__main__":
    # 请根据实际情况修改路径
    EN_DATA = "./datasets/mfd_generated_results.jsonl"
    ZH_DATA = "./datasets/cmfd_generated_results.jsonl"
    # 可以替换为您本地的模型路径或 HuggingFace ID
    MODEL = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B")  
    OUTPUT = "results/dual_lang_probe_experiment"
    
    experiment = DualLangProbeExperiment(EN_DATA, ZH_DATA, MODEL, OUTPUT)
    experiment.run()