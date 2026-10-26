import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 探针与潜空间管理器 (继承并优化了之前的稳健性补丁)
# =============================================================================
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class MoralSteeringManager:
    def __init__(self, model, tokenizer, target_layer, probe_path, vectors_dir, label_classes_path, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.device = device
        self.hidden_dim = model.config.hidden_size
        target_dtype = self.model.dtype # 确保精度对齐
        
        self.foundations = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
        
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
        
        # 加载道德向量并对齐精度
        self.vectors = {}
        for f in self.foundations:
            v_path = os.path.join(vectors_dir, f"layer_{self.target_layer}_{f.capitalize()}.pt")
            if not os.path.isfile(v_path):
                raise FileNotFoundError(v_path)
            if os.path.exists(v_path):
                v = torch.load(v_path, map_location=self.device)
                self.vectors[f] = v.view(1, 1, -1).to(target_dtype)
            else:
                self.vectors[f] = torch.zeros((1, 1, self.hidden_dim), device=self.device, dtype=target_dtype)
                
        self.mean_vector = torch.mean(torch.cat(list(self.vectors.values()), dim=0), dim=0, keepdim=True)
        
        # AMF 核心超参数 (对应论文设定)
        self.condition = "amf" 
        self.tau = 0.2          # Threshold 设定为 0.2
        self.lambda_base = 2.0  # Steering 基础系数
        
        self._register_hook()

    def _register_hook(self):
        def hook(module, inputs, outputs):
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
                        # 核心 AMF 计算: ReLU(P_vice - tau) * lambda
                        alpha = torch.relu(p_vice - self.tau) * self.lambda_base
                        fusion_vector += alpha.unsqueeze(-1) * self.vectors[f]
                        
                steered_hidden = hidden_states + fusion_vector
            
            if is_tuple:
                return (steered_hidden,) + outputs[1:]
            else:
                return steered_hidden

        self.hook_handle = self.model.model.layers[self.target_layer].register_forward_hook(hook)

    def set_condition(self, condition):
        self.condition = condition

    def remove_hook(self):
        self.hook_handle.remove()

# =============================================================================
# 2. 生成流程 (适配 Llama 3 Instruct Template)
# =============================================================================
def generate_response(model, tokenizer, prompt, device, max_new_tokens=150):
    # 1. 组合 message，并应用 chat template 生成纯文本 (tokenize=False)
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 2. 标准化 tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    # 3. 使用 **inputs 将字典解包传入 generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0, # Greedy decoding
            do_sample=False
        )
    
    # 4. 截取新生成的 token (扣除原 prompt 的长度)
    input_len = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    return response

# =============================================================================
# 3. 主程序
# =============================================================================
def main():
    # ---- 1. 环境与路径配置 ----
    MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    PROBE_PATH = "results/layerwise_probe_experiment/saved_probes/probe_MFD_layer_17.pt"
    VECTORS_DIR = "moral_vectors"
    LABEL_CLASSES_PATH = "results/layerwise_probe_experiment/saved_probes/label_classes.npy"
    TARGET_LAYER = 17
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---- 2. 加载模型与管理器 ----
    print("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    
    manager = MoralSteeringManager(
        model=model, tokenizer=tokenizer, target_layer=TARGET_LAYER,
        probe_path=PROBE_PATH, vectors_dir=VECTORS_DIR, 
        label_classes_path=LABEL_CLASSES_PATH, device=device
    )
    
    # 设置运行条件为 AMF
    manager.set_condition("amf")
    print(f"\n--- Current condition: {manager.condition.upper()} ---")

    # ---- 3. 处理 HarmBench 数据集 ----
    print("\nProcessing HarmBench Dataset...")
    harmbench_df = pd.read_csv("datasets/harmbench_behaviors_text_all.csv")
    hb_responses = []
    
    # 假设 prompt 列名为 'Behavior' (如不匹配请根据实际csv调整)
    prompt_col = 'Behavior' if 'Behavior' in harmbench_df.columns else harmbench_df.columns[0]
    
    for idx, row in tqdm(harmbench_df.iterrows(), total=len(harmbench_df)):
        prompt = row[prompt_col]
        resp = generate_response(model, tokenizer, prompt, device)
        hb_responses.append(resp)
        
    harmbench_df['generated_response'] = hb_responses
    harmbench_df.to_csv("harmbench_amf_results.csv", index=False)
    print("Saved HarmBench results to: harmbench_amf_results.csv")

    # ---- 4. 处理 XSTest 数据集 ----
    print("\nProcessing XSTest Dataset...")
    xstest_df = pd.read_csv("datasets/xstest_prompts.csv")
    xs_responses = []
    
    # 假设 prompt 列名为 'prompt' (如不匹配请根据实际csv调整)
    xs_prompt_col = 'prompt' if 'prompt' in xstest_df.columns else xstest_df.columns[0]
    
    for idx, row in tqdm(xstest_df.iterrows(), total=len(xstest_df)):
        prompt = row[xs_prompt_col]
        resp = generate_response(model, tokenizer, prompt, device)
        xs_responses.append(resp)
        
    xstest_df['generated_response'] = xs_responses
    xstest_df.to_csv("xstest_amf_results.csv", index=False)
    print("Saved XSTest results to: xstest_amf_results.csv")

    manager.remove_hook()
    print("\nGeneration Complete! You can now run your evaluation scripts on the output CSVs.")

if __name__ == "__main__":
    main()