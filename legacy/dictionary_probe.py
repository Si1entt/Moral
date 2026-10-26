"""
道德基础词典层级探针实验 (Layer-wise Probing with 10-class Classification)
假设：随着模型层数加深，道德概念的表征会发生变化。
我们在每一层训练一个10分类探针（5个道德基础 × 2个极性）

实验模型：Llama-3.1-8B
数据集：MFD (英文) 和 C-MFD (中文)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 数据加载模块
# =============================================================================

def load_mfd_data(mfd_path: str, cmfd_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载MFD和C-MFD数据集"""
    # 加载英文MFD
    mfd_df = pd.read_csv(mfd_path)
    mfd_df.columns = ['word', 'foundation']
    mfd_df['category'] = mfd_df['foundation'].apply(lambda x: x.split('.')[0])
    mfd_df['polarity'] = mfd_df['foundation'].apply(lambda x: x.split('.')[1])
    
    replacements = {'sanc': 'sanctity', 'auth': 'authority', 'loya': 'loyalty', 'fair': 'fairness'}
    mfd_df['category'] = mfd_df['category'].replace(replacements)
    
    # 加载中文C-MFD
    try:
        cmfd_df = pd.read_csv(cmfd_path, encoding='utf-8-sig')
    except:
        try:
            cmfd_df = pd.read_csv(cmfd_path, encoding='utf-8-sig')
        except:
            cmfd_df = pd.read_csv(cmfd_path, encoding='utf-8')

    cmfd_df.columns = ['word', 'foundation']
    cmfd_df['category'] = cmfd_df['foundation'].apply(lambda x: x.split('.')[0])
    cmfd_df['polarity'] = cmfd_df['foundation'].apply(lambda x: x.split('.')[1])
    cmfd_df['category'] = cmfd_df['category'].replace(replacements)

    print(f"MFD数据集: {len(mfd_df)} 个词")
    print(f"C-MFD数据集: {len(cmfd_df)} 个词")
    return mfd_df, cmfd_df


# =============================================================================
# 2. 模型加载和层级表征提取模块
# =============================================================================

class RepresentationExtractor:
    """提取每一层的表征"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        print(f"加载模型: {model_name}")
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
        print("模型加载完成")

    def extract_word_layers(self, word: str, language: str) -> np.ndarray:
        """提取单个词在所有层的表征向量"""
        if language == 'en':
            prompt = f"Your first instinct for the word \"{word}\"is: "
        else:
            prompt = f"你对”{word}“这个词的第一直觉是："
            
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        all_hidden_states = outputs.hidden_states
        layers_vec = np.stack([h[0, -1, :].cpu().float().numpy() for h in all_hidden_states])
        
        return layers_vec

    def extract_batch_representations(self, words: List[str], language: str = 'en') -> Dict[str, np.ndarray]:
        """批量提取，返回字典: word -> array(num_layers, hidden_dim)"""
        representations = {}
        for word in tqdm(words, desc=f"提取层级表征 ({language})"):
            try:
                repr_vec = self.extract_word_layers(word, language)
                representations[word] = repr_vec
            except Exception as e:
                continue
        return representations


# =============================================================================
# 3. 探针模型定义
# =============================================================================

class LinearProbe(nn.Module):
    """线性探针分类器"""
    def __init__(self, input_dim: int, num_classes: int = 10):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """多层感知机探针分类器"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 10):
        super(MLPProbe, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# 4. 探针训练器
# =============================================================================

class ProbeTrainer:
    """探针训练器"""
    
    def __init__(self, input_dim: int, num_classes: int = 10, 
                 probe_type: str = 'linear', device: str = 'cuda'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.probe_type = probe_type
        self.device = device
        
    def create_probe(self) -> nn.Module:
        """创建新的探针模型"""
        if self.probe_type == 'linear':
            return LinearProbe(self.input_dim, self.num_classes).to(self.device)
        else:
            return MLPProbe(self.input_dim, num_classes=self.num_classes).to(self.device)
    
    def train_probe(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 100, lr: float = 0.001, batch_size: int = 32,
                    verbose: bool = False) -> Tuple[nn.Module, Dict]:
        """
        训练探针
        返回: (训练好的模型, 训练历史)
        """
        probe = self.create_probe()
        
        # 转换为 PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        
        # 创建 DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            probe.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = probe(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            scheduler.step()
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(probe, X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        return probe, history
    
    def evaluate(self, probe: nn.Module, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """评估探针"""
        probe.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs = probe(X_t)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_t).item()
            _, predicted = outputs.max(1)
            acc = predicted.eq(y_t).sum().item() / len(y)
        
        return loss, acc
    
    def predict(self, probe: nn.Module, X: np.ndarray) -> np.ndarray:
        """预测"""
        probe.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = probe(X_t)
            _, predicted = outputs.max(1)
        
        return predicted.cpu().numpy()


# =============================================================================
# 5. 主实验流程
# =============================================================================

class MoralLayerProbeExperiment:
    """层级探针实验"""
    
    def __init__(self, mfd_path: str, cmfd_path: str, model_name: str,
                 probe_type: str = 'linear', n_folds: int = 5):
        self.mfd_df, self.cmfd_df = load_mfd_data(mfd_path, cmfd_path)
        self.model_name = model_name
        self.probe_type = probe_type
        self.n_folds = n_folds
        
        # 数据结构
        self.mfd_reprs = {}
        self.cmfd_reprs = {}
        
        # 10分类标签映射
        self.categories = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
        self.polarities = ['virtue', 'vice']
        
        # 创建标签编码器
        self.label_encoder = LabelEncoder()
        self.class_names = [f"{cat}.{pol}" for cat in self.categories for pol in self.polarities]
        self.label_encoder.fit(self.class_names)
        
        # 结果存储
        self.layer_results = []
        self.num_layers = 0
        self.hidden_dim = 0
        
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """为数据创建10分类标签"""
        labels = df['category'] + '.' + df['polarity']
        return labels
    
    def _prepare_data(self, df: pd.DataFrame, reprs: Dict[str, np.ndarray], 
                      layer_idx: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备指定层的训练数据
        返回: (X, y, words)
        """
        X_list = []
        y_list = []
        words = []
        
        for idx, row in df.iterrows():
            word = row['word']
            if word not in reprs:
                continue
            
            label = f"{row['category']}.{row['polarity']}"
            if label not in self.class_names:
                continue
            
            # 获取该层的表征
            vec = reprs[word][layer_idx]
            X_list.append(vec)
            y_list.append(self.label_encoder.transform([label])[0])
            words.append(word)
        
        return np.array(X_list), np.array(y_list), words
    
    # [修改] 增加 output_dir 参数，用于创建保存探针权重的文件夹
    def run(self, output_dir: str):
        """运行实验"""
        
        # [新增] 创建模型保存目录并保存类别映射文件
        probe_save_dir = os.path.join(output_dir, "saved_probes")
        os.makedirs(probe_save_dir, exist_ok=True)
        np.save(os.path.join(probe_save_dir, "label_classes.npy"), self.label_encoder.classes_)
        print(f"探针权重及类别映射将保存在: {probe_save_dir}")

        # 1. 提取表征
        extractor = RepresentationExtractor(self.model_name)
        
        self.mfd_reprs = extractor.extract_batch_representations(
            self.mfd_df['word'].tolist(), 'en')
        self.cmfd_reprs = extractor.extract_batch_representations(
            self.cmfd_df['word'].tolist(), 'zh')
        
        # 确定层数和维度
        if len(self.mfd_reprs) > 0:
            sample = list(self.mfd_reprs.values())[0]
            self.num_layers = sample.shape[0]
            self.hidden_dim = sample.shape[1]
        else:
            print("没有提取到表征!")
            return
        
        print(f"\n模型共 {self.num_layers} 层, 隐藏维度: {self.hidden_dim}")
        print(f"10分类标签: {self.class_names}")
        
        # 2. 逐层训练探针
        print(f"\n=== 开始逐层训练探针 (共 {self.num_layers} 层) ===")
        
        for layer_idx in tqdm(range(self.num_layers), desc="Training Layer Probes"):
            # [修改] 传递 probe_save_dir
            layer_res = self._train_layer_probe(layer_idx, probe_save_dir)
            self.layer_results.append(layer_res)
            
        print("\n实验完成!")
    
    # [修改] 接收 probe_save_dir 参数
    def _train_layer_probe(self, layer_idx: int, probe_save_dir: str) -> Dict:
        """训练单层的探针"""
        results = {'layer': layer_idx, 'MFD': {}, 'C-MFD': {}}
        
        for dataset_name, df, reprs in [('MFD', self.mfd_df, self.mfd_reprs),
                                         ('C-MFD', self.cmfd_df, self.cmfd_reprs)]:
            # 准备数据
            X, y, words = self._prepare_data(df, reprs, layer_idx)
            
            if len(X) < 20:
                print(f"警告: {dataset_name} 层{layer_idx} 数据量不足 ({len(X)})")
                results[dataset_name] = {
                    'accuracy': np.nan,
                    'accuracy_std': np.nan,
                    'per_class_accuracy': {},
                    'confusion_matrix': None
                }
                continue
            
            # 使用交叉验证
            trainer = ProbeTrainer(
                input_dim=self.hidden_dim,
                num_classes=len(self.class_names),
                probe_type=self.probe_type,
                device=self.device
            )
            
            fold_accuracies = []
            all_preds = []
            all_labels = []
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 训练探针
                probe, history = trainer.train_probe(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=100, lr=0.001, batch_size=256,
                    verbose=False
                )
                
                # 评估
                _, acc = trainer.evaluate(probe, X_test, y_test)
                fold_accuracies.append(acc)
                
                # 收集预测结果用于混淆矩阵
                preds = trainer.predict(probe, X_test)
                all_preds.extend(preds)
                all_labels.extend(y_test)
            
            # [新增] 在交叉验证完成后，使用该层该数据集的全量数据 (X, y) 训练最终探针并保存
            final_probe, _ = trainer.train_probe(
                X, y,
                epochs=100, lr=0.001, batch_size=256,
                verbose=False
            )
            
            # 构建保存路径并保存权重
            save_path = os.path.join(probe_save_dir, f"probe_{dataset_name}_layer_{layer_idx}.pt")
            torch.save(final_probe.state_dict(), save_path)
            
            # 计算每类准确率
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            per_class_acc = {}
            for i, class_name in enumerate(self.class_names):
                mask = all_labels == i
                if mask.sum() > 0:
                    per_class_acc[class_name] = (all_preds[mask] == all_labels[mask]).mean()
                else:
                    per_class_acc[class_name] = np.nan
            
            # 混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            
            results[dataset_name] = {
                'accuracy': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'per_class_accuracy': per_class_acc,
                'confusion_matrix': cm
            }
        
        return results
    
    def visualize(self, output_dir: str):
        """可视化结果"""
        if not self.layer_results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 准备绘图数据
        plot_data = []
        for r in self.layer_results:
            layer = r['layer']
            for ds in ['MFD', 'C-MFD']:
                acc = r[ds].get('accuracy', np.nan)
                acc_std = r[ds].get('accuracy_std', np.nan)
                plot_data.append({
                    'Layer': layer,
                    'Dataset': ds,
                    'Accuracy': acc,
                    'Accuracy_Std': acc_std
                })
        
        df = pd.DataFrame(plot_data)
        
        # 图1: 各层探针准确率
        plt.figure(figsize=(14, 6))
        for ds in ['MFD', 'C-MFD']:
            ds_df = df[df['Dataset'] == ds]
            plt.plot(ds_df['Layer'], ds_df['Accuracy'], marker='o', label=ds, linewidth=2)
            plt.fill_between(
                ds_df['Layer'],
                ds_df['Accuracy'] - ds_df['Accuracy_Std'],
                ds_df['Accuracy'] + ds_df['Accuracy_Std'],
                alpha=0.2
            )
        
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('10分类探针准确率随层数变化 (5-Fold CV)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.1, color='r', linestyle='--', label='Random (10%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_probe_accuracy.png'), dpi=150)
        plt.show()
        
        # 图2: 每类准确率热力图 (选择几个关键层)
        key_layers = [0, self.num_layers // 4, self.num_layers // 2, 
                      3 * self.num_layers // 4, self.num_layers - 1]
        
        for ds in ['MFD', 'C-MFD']:
            fig, axes = plt.subplots(1, len(key_layers), figsize=(20, 5))
            
            for i, layer_idx in enumerate(key_layers):
                if layer_idx >= len(self.layer_results):
                    continue
                    
                per_class = self.layer_results[layer_idx][ds].get('per_class_accuracy', {})
                if not per_class:
                    continue
                
                # 转换为矩阵形式 (5 categories x 2 polarities)
                matrix = np.zeros((5, 2))
                for cat_idx, cat in enumerate(self.categories):
                    for pol_idx, pol in enumerate(self.polarities):
                        label = f"{cat}.{pol}"
                        matrix[cat_idx, pol_idx] = per_class.get(label, np.nan)
                
                sns.heatmap(matrix, ax=axes[i], annot=True, fmt='.2f',
                           xticklabels=self.polarities, yticklabels=self.categories,
                           cmap='YlGnBu', vmin=0, vmax=1)
                axes[i].set_title(f'Layer {layer_idx}')
            
            plt.suptitle(f'{ds}: 各类别准确率 (关键层)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'per_class_accuracy_{ds}.png'), dpi=150)
            plt.show()
        
        # 图3: 最后一层的混淆矩阵
        for ds in ['MFD', 'C-MFD']:
            cm = self.layer_results[-1][ds].get('confusion_matrix')
            if cm is None:
                continue
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{ds}: 最后一层混淆矩阵')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{ds}_last_layer.png'), dpi=150)
            plt.show()
        
        # 图4: Virtue vs Vice 准确率对比
        virtue_accs = {ds: [] for ds in ['MFD', 'C-MFD']}
        vice_accs = {ds: [] for ds in ['MFD', 'C-MFD']}
        
        for r in self.layer_results:
            for ds in ['MFD', 'C-MFD']:
                per_class = r[ds].get('per_class_accuracy', {})
                if not per_class:
                    virtue_accs[ds].append(np.nan)
                    vice_accs[ds].append(np.nan)
                    continue
                
                virtue_vals = [per_class.get(f"{cat}.virtue", np.nan) for cat in self.categories]
                vice_vals = [per_class.get(f"{cat}.vice", np.nan) for cat in self.categories]
                
                virtue_accs[ds].append(np.nanmean(virtue_vals))
                vice_accs[ds].append(np.nanmean(vice_vals))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        layers = range(self.num_layers)
        
        for i, ds in enumerate(['MFD', 'C-MFD']):
            axes[i].plot(layers, virtue_accs[ds], marker='o', label='Virtue', linewidth=2)
            axes[i].plot(layers, vice_accs[ds], marker='s', label='Vice', linewidth=2)
            axes[i].set_xlabel('Layer')
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(f'{ds}: Virtue vs Vice 分类准确率')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'virtue_vs_vice_accuracy.png'), dpi=150)
        plt.show()
        
        # 保存详细结果到CSV
        detailed_results = []
        for r in self.layer_results:
            layer = r['layer']
            for ds in ['MFD', 'C-MFD']:
                row = {
                    'Layer': layer,
                    'Dataset': ds,
                    'Overall_Accuracy': r[ds].get('accuracy', np.nan),
                    'Accuracy_Std': r[ds].get('accuracy_std', np.nan)
                }
                per_class = r[ds].get('per_class_accuracy', {})
                for class_name in self.class_names:
                    row[f'Acc_{class_name}'] = per_class.get(class_name, np.nan)
                detailed_results.append(row)
        
        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv(os.path.join(output_dir, 'layerwise_probe_results.csv'), index=False)
        
        print(f"\n结果已保存至 {output_dir}")
        
        # 打印最佳层
        print("\n=== 最佳探针层 ===")
        for ds in ['MFD', 'C-MFD']:
            ds_df = results_df[results_df['Dataset'] == ds]
            best_idx = ds_df['Overall_Accuracy'].idxmax()
            best_layer = ds_df.loc[best_idx, 'Layer']
            best_acc = ds_df.loc[best_idx, 'Overall_Accuracy']
            print(f"{ds}: 最佳层 = {best_layer}, 准确率 = {best_acc:.4f}")


# =============================================================================
# 6. 主程序
# =============================================================================

def main():
    MFD_PATH = "datasets/MFD.csv"
    CMFD_PATH = "datasets/C-MFD.csv"
    MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    OUTPUT_DIR = "results/layerwise_probe_experiment"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    exp = MoralLayerProbeExperiment(
        mfd_path=MFD_PATH,
        cmfd_path=CMFD_PATH,
        model_name=MODEL_PATH,
        probe_type='linear',  # 可选 'linear' 或 'mlp'
        n_folds=10
    )
    
    # [修改] 传入 OUTPUT_DIR 参数
    exp.run(OUTPUT_DIR)
    exp.visualize(OUTPUT_DIR)


if __name__ == "__main__":
    main()