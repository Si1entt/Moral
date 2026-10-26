"""
效用基准测试 (Utility Benchmarks Evaluation)
用于 Rebuttal 补充实验：评估 Baseline, Mean Vector, 和 AMF 三种条件对模型通用能力的影响。
评估数据集: ARC-Challenge, SciQ, MATH
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 探针定义 (与原 probe.py 保持一致)
# =============================================================================
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# =============================================================================
# 2. 潜空间干预管理器 (Activation Steering Manager)
# =============================================================================
class MoralSteeringManager:
    def __init__(self, model, tokenizer, target_layer, probe_path, vectors_dir, label_classes_path, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device
        self.hidden_dim = model.config.hidden_size
        
        self.foundations = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
        
        # 加载类别映射
        if not os.path.isfile(label_classes_path):
            raise FileNotFoundError(label_classes_path)
        if os.path.exists(label_classes_path):
            self.class_names = np.load(label_classes_path, allow_pickle=True).tolist()
        else:
            polarities = ['virtue', 'vice']
            self.class_names = [f"{cat}.{pol}" for cat in self.foundations for pol in polarities]
            
        # 预计算 Vice 索引，用于 AMF 提取概率
        self.vice_indices = {f: self.class_names.index(f"{f}.vice") for f in self.foundations}

        # 获取模型的数据类型 (通常是 torch.float16)
        target_dtype = self.model.dtype

        # 加载探针并对齐数据类型
        self.probe = LinearProbe(self.hidden_dim, len(self.class_names)).to(self.device).to(target_dtype)
        if not os.path.isfile(probe_path):
            raise FileNotFoundError(probe_path)
        if os.path.exists(probe_path):
            self.probe.load_state_dict(torch.load(probe_path, map_location=self.device))
        self.probe.eval()
        
        # 加载道德向量并对齐数据类型
        self.vectors = {}
        for f in self.foundations:
            v_path = os.path.join(vectors_dir, f"layer_{self.target_layer}_{f.capitalize()}.pt")
            if not os.path.isfile(v_path):
                raise FileNotFoundError(v_path)
            if os.path.exists(v_path):
                v = torch.load(v_path, map_location=self.device)
                self.vectors[f] = v.view(1, 1, -1).to(target_dtype) # 确保类型一致
            else:
                print(f"警告: 找不到向量文件 {v_path}，将使用零向量代替。")
                self.vectors[f] = torch.zeros((1, 1, self.hidden_dim), device=self.device, dtype=target_dtype)
                
        # 计算 Mean Vector (类型会自然继承)
        self.mean_vector = torch.mean(torch.cat(list(self.vectors.values()), dim=0), dim=0, keepdim=True)
        
        self.condition = "baseline" # 可选: 'baseline', 'mean', 'amf'
        self.tau = 0.2
        self.lambda_base = 2.0
        
        self._register_hook()

    def _register_hook(self):
        """在目标层注册 forward hook 以干预隐藏状态"""
        def hook(module, inputs, outputs):
            # 动态检查 outputs 类型：可能是 tuple，也可能是单个 Tensor
            is_tuple = isinstance(outputs, tuple)
            hidden_states = outputs[0] if is_tuple else outputs
            
            if self.condition == "baseline":
                steered_hidden = hidden_states
                
            elif self.condition == "mean":
                steered_hidden = hidden_states + self.lambda_base * self.mean_vector
                
            elif self.condition == "amf":
                with torch.no_grad():
                    logits = self.probe(hidden_states)
                    probs = torch.softmax(logits, dim=-1)
                    
                    fusion_vector = torch.zeros_like(hidden_states)
                    for f in self.foundations:
                        p_vice = probs[..., self.vice_indices[f]]
                        alpha = torch.relu(p_vice - self.tau) * self.lambda_base
                        fusion_vector += alpha.unsqueeze(-1) * self.vectors[f]
                        
                steered_hidden = hidden_states + fusion_vector
            
            # 保持与原 outputs 相同的返回结构
            if is_tuple:
                return (steered_hidden,) + outputs[1:]
            else:
                return steered_hidden

        # 挂载到 Llama 的指定层
        self.hook_handle = self.model.model.layers[self.target_layer].register_forward_hook(hook)

    def set_condition(self, condition):
        assert condition in ["baseline", "mean", "amf"]
        self.condition = condition

    def remove_hook(self):
        self.hook_handle.remove()

# =============================================================================
# 3. 评测数据集构造器与评估函数
# =============================================================================
def generate_answer(model, tokenizer, prompt, device, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0, # Greedy decoding
            do_sample=False
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def evaluate_sciq(model, tokenizer, device, num_samples=200):
    print("Evaluating SciQ...")
    dataset = load_dataset("sciq", split=(f"validation[:{num_samples}]" if num_samples is not None and num_samples > 0 else "validation"))
    correct = 0
    
    for item in tqdm(dataset):
        prompt = f"Question: {item['question']}\nProvide the exact correct answer:\nAnswer:"
        pred = generate_answer(model, tokenizer, prompt, device).lower()
        if item['correct_answer'].lower() in pred:
            correct += 1
            
    return correct / len(dataset)

def evaluate_arc(model, tokenizer, device, num_samples=200):
    print("Evaluating ARC-Challenge...")
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split=(f"validation[:{num_samples}]" if num_samples is not None and num_samples > 0 else "validation"))
    correct = 0
    
    for item in tqdm(dataset):
        choices = item['choices']
        labels = choices['label']
        texts = choices['text']
        
        options_text = "\n".join([f"{lbl}) {txt}" for lbl, txt in zip(labels, texts)])
        prompt = f"Question: {item['question']}\n{options_text}\nWhich is the correct option? Only output the option letter.\nAnswer:"
        
        pred = generate_answer(model, tokenizer, prompt, device, max_new_tokens=5).strip()
        
        # 提取首个有效字符作为预测选项
        pred_label = ""
        for char in pred:
            if char in labels:
                pred_label = char
                break
                
        if pred_label == item['answerKey']:
            correct += 1
            
    return correct / len(dataset)

def evaluate_math(model, tokenizer, device, num_samples=None):
    print("Evaluating MATH (HuggingFaceH4/MATH-500)...")
    
    # 使用无需登录验证、开箱即用的优质 MATH 测试集子集
    split_name = f"test[:{num_samples}]" if num_samples is not None and num_samples > 0 else "test"
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split_name)
        
    correct = 0
    for item in tqdm(dataset):
        # 构造简单的 Prompt 诱导模型输出
        prompt = f"Solve the following math problem. Only output the final numerical answer.\nProblem: {item['problem']}\nFinal Answer:"
        
        # MATH 题目的输出生成可以稍微放宽一点 token 限制
        pred = generate_answer(model, tokenizer, prompt, device, max_new_tokens=25)
        
        # MATH-500 数据集直接提供了提取好的 exact answer
        ans_str = str(item['answer']).strip()
        
        # 启发式包含匹配
        if ans_str in pred:
            correct += 1
            
    return correct / len(dataset)

# =============================================================================
# 4. 主程序
# =============================================================================
def main():
    # ---- 路径配置 (请替换为你的实际路径) ----
    MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    PROBE_PATH = "results/layerwise_probe_experiment/saved_probes/probe_MFD_layer_17.pt"
    VECTORS_DIR = "moral_vectors"
    LABEL_CLASSES_PATH = "results/layerwise_probe_experiment/saved_probes/label_classes.npy"
    TARGET_LAYER = 17
    EVAL_SAMPLES = -1 # 为了快速跑通设为100，正式写入论文建议设为全集或至少500
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 加载模型 ----
    print(f"Loading Model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 初始化 Steering Manager ----
    print("Initializing Steering Manager...")
    manager = MoralSteeringManager(
        model=model, 
        tokenizer=tokenizer, 
        target_layer=TARGET_LAYER,
        probe_path=PROBE_PATH,
        vectors_dir=VECTORS_DIR,
        label_classes_path=LABEL_CLASSES_PATH,
        device=device
    )

    conditions = ["baseline", "mean", "amf"]
    results = {
        "Condition": conditions,
        "ARC (Acc)": [],
        "SciQ (Acc)": [],
        "MATH (Acc)": []
    }

    # ---- 开始评测 ----
    for cond in conditions:
        print(f"\n{'='*40}")
        print(f"Running Evaluation for Condition: {cond.upper()}")
        print(f"{'='*40}")
        
        manager.set_condition(cond)
        
        acc_arc = evaluate_arc(model, tokenizer, device, num_samples=EVAL_SAMPLES)
        acc_sciq = evaluate_sciq(model, tokenizer, device, num_samples=EVAL_SAMPLES)
        acc_math = evaluate_math(model, tokenizer, device, num_samples=EVAL_SAMPLES)
        
        results["ARC (Acc)"].append(f"{acc_arc*100:.2f}%")
        results["SciQ (Acc)"].append(f"{acc_sciq*100:.2f}%")
        results["MATH (Acc)"].append(f"{acc_math*100:.2f}%")

    # ---- 清理 Hook 并输出结果 ----
    manager.remove_hook()
    
    df_results = pd.DataFrame(results)
    print("\n\n" + "="*50)
    print("FINAL UTILITY BENCHMARK RESULTS")
    print("="*50)
    print(df_results.to_markdown(index=False))
    
    # 存为 CSV 以便放入论文
    df_results.to_csv("utility_eval_results.csv", index=False)
    print("\nResults saved to utility_eval_results.csv")

if __name__ == "__main__":
    main()