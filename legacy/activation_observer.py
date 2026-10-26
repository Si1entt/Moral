import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')

# 1. 探针定义 (与 generate.py 保持一致)
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# 2. 探针观测器管理器 (只观测，不修改 Hidden States)
class ActivationObserver:
    def __init__(self, model, target_layer, probe_path, label_classes_path, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.hidden_dim = model.config.hidden_size
        target_dtype = self.model.dtype 
        
        self.foundations = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
        self.tau = 0.2  # 与 AMF 保持一致的激活阈值
        
        # 加载类别映射
        import numpy as np
        if not os.path.isfile(label_classes_path):
            raise FileNotFoundError(label_classes_path)
        if os.path.exists(label_classes_path):
            self.class_names = np.load(label_classes_path, allow_pickle=True).tolist()
        else:
            polarities = ['virtue', 'vice']
            self.class_names = [f"{cat}.{pol}" for cat in self.foundations for pol in polarities]
            
        self.vice_indices = {f: self.class_names.index(f"{f}.vice") for f in self.foundations}

        # 加载探针并对齐精度
        self.probe = LinearProbe(self.hidden_dim, len(self.class_names)).to(self.device).to(target_dtype)
        if not os.path.isfile(probe_path):
            raise FileNotFoundError(probe_path)
        if os.path.exists(probe_path):
            self.probe.load_state_dict(torch.load(probe_path, map_location=self.device))
        self.probe.eval()
        
        self.current_stats = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, inputs, outputs):
            is_tuple = isinstance(outputs, tuple)
            hidden_states = outputs[0] if is_tuple else outputs
            
            # 我们只关心 Prompt 最后一个 token 的激活状态
            last_token_hidden = hidden_states[:, -1, :]
            
            with torch.no_grad():
                logits = self.probe(last_token_hidden)
                probs = torch.softmax(logits, dim=-1)
                
                stats = {}
                activated_count = 0
                for f in self.foundations:
                    # 获取当前维度的 vice 概率
                    p_vice = probs[0, self.vice_indices[f]].item()
                    is_activated = bool(p_vice > self.tau)
                    
                    stats[f] = {
                        "p_vice": round(p_vice, 4),
                        "activated": is_activated
                    }
                    if is_activated:
                        activated_count += 1
                
                stats["total_activated"] = activated_count
                self.current_stats = stats
                
            return outputs # 不修改输出，纯观测

        self.hook_handle = self.model.model.layers[self.target_layer].register_forward_hook(hook)

    def remove_hook(self):
        self.hook_handle.remove()

# 3. 主程序逻辑
def main():
    # ---- 请根据你服务器的实际路径修改 ----
    MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")  
    PROBE_PATH = "results/layerwise_probe_experiment/saved_probes/probe_MFD_layer_17.pt"
    LABEL_CLASSES_PATH = "results/layerwise_probe_experiment/saved_probes/label_classes.npy"
    TARGET_LAYER = 17
    CSV_PATH = "datasets/harmbench_behaviors_text_all.csv" # 原始 Harmbench 数据集
    # -------------------------------------
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    
    observer = ActivationObserver(model, TARGET_LAYER, PROBE_PATH, LABEL_CLASSES_PATH, device)
    
    print(f"Loading dataset {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset {CSV_PATH} not found.")
        return
        
    prompt_col = 'Behavior' if 'Behavior' in df.columns else df.columns[0]
    
    results = []
    
    print("Analyzing mechanistic activations (Forward Pass Only)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row[prompt_col]
        
        # 组装 Chat Template
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        
        # 仅仅做一次 Forward 触发 Hook，不需要 generate() 文本
        with torch.no_grad():
            _ = model(**inputs)
            
        stats = observer.current_stats
        
        # 记录每条结果
        record = {
            "Behavior": prompt,
            "total_activated": stats["total_activated"]
        }
        for f in observer.foundations:
            record[f"{f}_p_vice"] = stats[f]["p_vice"]
            record[f"{f}_activated"] = stats[f]["activated"]
            
        results.append(record)
        
    # 分析和保存统计数据
    results_df = pd.DataFrame(results)
    
    os.makedirs("results/mechanistic", exist_ok=True)
    save_path = "results/mechanistic/harmbench_activation_stats.csv"
    results_df.to_csv(save_path, index=False)
    
    # 打印给 Reviewer 看的统计结论
    print("\n" + "="*60)
    print("🔬 Mechanistic Analysis: Moral Dimension Activations (tau=0.2)")
    print("="*60)
    
    # 1. 独立维度的激活频率
    print("\n[1] Frequency of individual moral foundations being activated:")
    for f in observer.foundations:
        act_rate = results_df[f"{f}_activated"].mean() * 100
        print(f"  - {f.capitalize():<10}: {act_rate:>5.2f}%")
        
    # 2. 同时激活维度数量的分布
    print("\n[2] Distribution of simultaneous activations (0 to 5):")
    counts = results_df['total_activated'].value_counts().sort_index()
    total_samples = len(results_df)
    for i in range(6):
        count = counts.get(i, 0)
        percentage = (count / total_samples) * 100
        print(f"  - {i} foundations activated: {count:>4} prompts ({percentage:>5.2f}%)")
        
    print("\n" + "-"*60)
    print("💡 Conclusion for Rebuttal:")
    print("If most prompts activate only 1-2 dimensions, a Mean Steering")
    print("Vector (which shifts all 5 dimensions equally) clearly causes")
    print("OVER-STEERING. Adaptive Moral Fusion solves this by only")
    print("steering the exactly activated dimensions.")
    print(f"Raw data saved to: {save_path}")

if __name__ == "__main__":
    main()