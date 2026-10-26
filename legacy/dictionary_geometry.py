"""
道德基础词典表征实验
假设：表达"好"的道德词汇的表征向量比较集中，但表征"坏"的表征向量比较分散

实验模型：Llama-3.1-8B
数据集：MFD (英文) 和 C-MFD (中文)
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
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

    # 解析foundation列
    mfd_df['category'] = mfd_df['foundation'].apply(lambda x: x.split('.')[0])
    mfd_df['polarity'] = mfd_df['foundation'].apply(lambda x: x.split('.')[1])

    # 统一类别名称 (sanc -> sanctity)
    mfd_df['category'] = mfd_df['category'].replace({'sanc': 'sanctity'})

    # 加载中文C-MFD (尝试不同编码)
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
    cmfd_df['category'] = cmfd_df['category'].replace({'sanc': 'sanctity'})

    print(f"MFD数据集加载完成: {len(mfd_df)} 个词")
    print(f"C-MFD数据集加载完成: {len(cmfd_df)} 个词")

    # 打印各类别统计
    print("\nMFD类别统计:")
    print(mfd_df.groupby(['category', 'polarity']).size().unstack(fill_value=0))
    print("\nC-MFD类别统计:")
    print(cmfd_df.groupby(['category', 'polarity']).size().unstack(fill_value=0))

    return mfd_df, cmfd_df


# =============================================================================
# 2. 模型加载和表征提取模块
# =============================================================================

class RepresentationExtractor:
    """使用Llama-3.1-8B提取词汇表征"""

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

        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("模型加载完成")

    def extract_word_representation(self, word: str) -> np.ndarray:
        """提取单个词的最后一层表征向量"""
        # 对词进行编码
        inputs = self.tokenizer(word, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

        # 取最后一个token的表征（或者取平均）
        # 这里使用最后一个有效token的表征
        word_repr = last_hidden_state[0, -1, :].cpu().numpy()

        return word_repr

    def extract_batch_representations(self, words: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """批量提取词汇表征"""
        representations = {}

        for i in tqdm(range(0, len(words), batch_size), desc="提取表征"):
            batch_words = words[i:i + batch_size]

            for word in batch_words:
                try:
                    repr_vec = self.extract_word_representation(word)
                    representations[word] = repr_vec
                except Exception as e:
                    print(f"处理词 '{word}' 时出错: {e}")
                    continue

        return representations


# =============================================================================
# 3. Whitening处理模块
# =============================================================================

def compute_whitening_params(representations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算Whitening参数"""
    # 计算均值
    mean = np.mean(representations, axis=0)

    # 中心化
    centered = representations - mean

    # 计算协方差矩阵
    cov = np.cov(centered.T)

    # SVD分解
    U, S, Vt = np.linalg.svd(cov)

    # 计算变换矩阵
    W = np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-8)))

    return mean, W


def apply_whitening(representations: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    """应用Whitening变换"""
    centered = representations - mean
    whitened = np.dot(centered, W)
    return whitened


# =============================================================================
# 4. 分析计算模块
# =============================================================================

def compute_intra_class_dispersion(vectors: np.ndarray) -> Dict[str, float]:
    """计算类内离散度的多种指标"""
    if len(vectors) < 2:
        return {'avg_distance': np.nan, 'std_distance': np.nan, 'avg_cosine_sim': np.nan}

    # 计算欧氏距离
    distances = pdist(vectors, metric='euclidean')

    # 计算余弦相似度
    cos_sim_matrix = cosine_similarity(vectors)
    # 取上三角（不包括对角线）
    upper_tri_indices = np.triu_indices(len(vectors), k=1)
    cos_similarities = cos_sim_matrix[upper_tri_indices]

    return {
        'avg_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'avg_cosine_sim': np.mean(cos_similarities),
        'std_cosine_sim': np.std(cos_similarities),
        'num_samples': len(vectors)
    }


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """计算类中心向量"""
    return np.mean(vectors, axis=0)


def compute_direction_relationship(virtue_centroid: np.ndarray, vice_centroid: np.ndarray) -> Dict[str, float]:
    """计算virtue和vice方向的关系"""
    # 余弦相似度
    cos_sim = cosine_similarity([virtue_centroid], [vice_centroid])[0, 0]

    # 欧氏距离
    euclidean_dist = np.linalg.norm(virtue_centroid - vice_centroid)

    # 向量的模
    virtue_norm = np.linalg.norm(virtue_centroid)
    vice_norm = np.linalg.norm(vice_centroid)

    return {
        'cosine_similarity': cos_sim,
        'euclidean_distance': euclidean_dist,
        'virtue_norm': virtue_norm,
        'vice_norm': vice_norm,
        'is_opposite': cos_sim < -0.5,  # 判断是否相反
        'is_orthogonal': abs(cos_sim) < 0.1  # 判断是否正交
    }


def compute_cross_dataset_similarity(
        mfd_vectors: np.ndarray,
        cmfd_vectors: np.ndarray
) -> Dict[str, float]:
    """计算跨数据集的相似度"""
    mfd_centroid = compute_centroid(mfd_vectors)
    cmfd_centroid = compute_centroid(cmfd_vectors)

    # 中心之间的相似度
    centroid_cos_sim = cosine_similarity([mfd_centroid], [cmfd_centroid])[0, 0]
    centroid_distance = np.linalg.norm(mfd_centroid - cmfd_centroid)

    return {
        'centroid_cosine_similarity': centroid_cos_sim,
        'centroid_euclidean_distance': centroid_distance
    }


# =============================================================================
# 5. 主实验流程
# =============================================================================

class MoralRepresentationExperiment:
    """道德表征实验主类"""

    def __init__(self, mfd_path: str, cmfd_path: str, model_name: str = "meta-llama/Llama-3.1-8B"):
        self.mfd_path = mfd_path
        self.cmfd_path = cmfd_path
        self.model_name = model_name

        # 加载数据
        self.mfd_df, self.cmfd_df = load_mfd_data(mfd_path, cmfd_path)

        # 存储结果
        self.mfd_representations = {}
        self.cmfd_representations = {}
        self.results = {}

        # 定义类别
        self.categories = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
        self.polarities = ['virtue', 'vice']

    def extract_all_representations(self, use_whitening: bool = True):
        """提取所有词汇的表征"""
        extractor = RepresentationExtractor(self.model_name)

        # 提取MFD表征
        print("\n=== 提取MFD (英文) 表征 ===")
        mfd_words = self.mfd_df['word'].tolist()
        self.mfd_representations = extractor.extract_batch_representations(mfd_words)

        # 提取C-MFD表征
        print("\n=== 提取C-MFD (中文) 表征 ===")
        cmfd_words = self.cmfd_df['word'].tolist()
        self.cmfd_representations = extractor.extract_batch_representations(cmfd_words)

        # 可选：应用Whitening
        if use_whitening:
            print("\n=== 应用Whitening变换 ===")
            self._apply_whitening()

    def _apply_whitening(self):
        """对表征应用Whitening"""
        # MFD Whitening
        mfd_vectors = np.array(list(self.mfd_representations.values()))
        mfd_mean, mfd_W = compute_whitening_params(mfd_vectors)
        mfd_whitened = apply_whitening(mfd_vectors, mfd_mean, mfd_W)

        for i, word in enumerate(self.mfd_representations.keys()):
            self.mfd_representations[word] = mfd_whitened[i]

        # C-MFD Whitening
        cmfd_vectors = np.array(list(self.cmfd_representations.values()))
        cmfd_mean, cmfd_W = compute_whitening_params(cmfd_vectors)
        cmfd_whitened = apply_whitening(cmfd_vectors, cmfd_mean, cmfd_W)

        for i, word in enumerate(self.cmfd_representations.keys()):
            self.cmfd_representations[word] = cmfd_whitened[i]

        print("Whitening完成")

    def _get_category_vectors(self, df: pd.DataFrame, representations: Dict[str, np.ndarray],
                              category: str, polarity: str) -> np.ndarray:
        """获取特定类别和极性的向量"""
        mask = (df['category'] == category) & (df['polarity'] == polarity)
        words = df[mask]['word'].tolist()
        vectors = [representations[w] for w in words if w in representations]
        return np.array(vectors) if vectors else np.array([])

    def analyze_intra_class_dispersion(self):
        """分析类内离散度"""
        print("\n=== 分析类内离散度 ===")

        results = {'MFD': {}, 'C-MFD': {}}

        for dataset_name, df, representations in [
            ('MFD', self.mfd_df, self.mfd_representations),
            ('C-MFD', self.cmfd_df, self.cmfd_representations)
        ]:
            print(f"\n{dataset_name}:")
            for category in self.categories:
                results[dataset_name][category] = {}
                for polarity in self.polarities:
                    vectors = self._get_category_vectors(df, representations, category, polarity)
                    if len(vectors) > 0:
                        dispersion = compute_intra_class_dispersion(vectors)
                        results[dataset_name][category][polarity] = dispersion
                        print(f"  {category}.{polarity}: "
                              f"avg_cos_sim={dispersion['avg_cosine_sim']:.4f}, "
                              f"n={dispersion['num_samples']}")

        self.results['intra_class_dispersion'] = results
        return results

    def analyze_virtue_vice_direction(self):
        """分析virtue和vice的方向关系"""
        print("\n=== 分析Virtue与Vice方向关系 ===")

        results = {'MFD': {}, 'C-MFD': {}}

        for dataset_name, df, representations in [
            ('MFD', self.mfd_df, self.mfd_representations),
            ('C-MFD', self.cmfd_df, self.cmfd_representations)
        ]:
            print(f"\n{dataset_name}:")
            for category in self.categories:
                virtue_vectors = self._get_category_vectors(df, representations, category, 'virtue')
                vice_vectors = self._get_category_vectors(df, representations, category, 'vice')

                if len(virtue_vectors) > 0 and len(vice_vectors) > 0:
                    virtue_centroid = compute_centroid(virtue_vectors)
                    vice_centroid = compute_centroid(vice_vectors)

                    relationship = compute_direction_relationship(virtue_centroid, vice_centroid)
                    results[dataset_name][category] = relationship

                    print(f"  {category}: cos_sim={relationship['cosine_similarity']:.4f}, "
                          f"opposite={relationship['is_opposite']}, "
                          f"orthogonal={relationship['is_orthogonal']}")

        self.results['virtue_vice_direction'] = results
        return results

    def analyze_cross_dataset_similarity(self):
        """分析跨数据集的相似度"""
        print("\n=== 分析跨数据集相似度 ===")

        results = {}

        for category in self.categories:
            results[category] = {}
            for polarity in self.polarities:
                mfd_vectors = self._get_category_vectors(
                    self.mfd_df, self.mfd_representations, category, polarity)
                cmfd_vectors = self._get_category_vectors(
                    self.cmfd_df, self.cmfd_representations, category, polarity)

                if len(mfd_vectors) > 0 and len(cmfd_vectors) > 0:
                    similarity = compute_cross_dataset_similarity(mfd_vectors, cmfd_vectors)
                    results[category][polarity] = similarity
                    print(f"  {category}.{polarity}: "
                          f"centroid_cos_sim={similarity['centroid_cosine_similarity']:.4f}")

        self.results['cross_dataset_similarity'] = results
        return results

    def compare_virtue_vice_dispersion(self):
        """比较virtue和vice的离散度差异"""
        print("\n=== Virtue vs Vice 离散度对比 ===")

        comparison = []

        for dataset_name in ['MFD', 'C-MFD']:
            for category in self.categories:
                try:
                    virtue_disp = self.results['intra_class_dispersion'][dataset_name][category]['virtue']
                    vice_disp = self.results['intra_class_dispersion'][dataset_name][category]['vice']

                    # virtue更集中意味着更高的cosine相似度
                    virtue_more_concentrated = virtue_disp['avg_cosine_sim'] > vice_disp['avg_cosine_sim']

                    comparison.append({
                        'dataset': dataset_name,
                        'category': category,
                        'virtue_avg_cos_sim': virtue_disp['avg_cosine_sim'],
                        'vice_avg_cos_sim': vice_disp['avg_cosine_sim'],
                        'difference': virtue_disp['avg_cosine_sim'] - vice_disp['avg_cosine_sim'],
                        'virtue_more_concentrated': virtue_more_concentrated
                    })

                    status = "✓ 支持假设" if virtue_more_concentrated else "✗ 不支持假设"
                    print(f"  {dataset_name}/{category}: virtue={virtue_disp['avg_cosine_sim']:.4f}, "
                          f"vice={vice_disp['avg_cosine_sim']:.4f} -> {status}")
                except KeyError:
                    continue

        comparison_df = pd.DataFrame(comparison)
        self.results['virtue_vice_comparison'] = comparison_df

        # 统计支持假设的比例
        support_ratio = comparison_df['virtue_more_concentrated'].mean()
        print(f"\n假设支持率: {support_ratio * 100:.1f}%")

        return comparison_df

    def run_full_experiment(self, use_whitening: bool = True):
        """运行完整实验"""
        print("=" * 60)
        print("道德基础词典表征实验")
        print("=" * 60)

        # Step 1: 提取表征
        self.extract_all_representations(use_whitening=use_whitening)

        # Step 2: 分析类内离散度
        self.analyze_intra_class_dispersion()

        # Step 3: 分析virtue和vice方向关系
        self.analyze_virtue_vice_direction()

        # Step 4: 分析跨数据集相似度
        self.analyze_cross_dataset_similarity()

        # Step 5: 对比分析
        self.compare_virtue_vice_dispersion()

        return self.results

    def visualize_results(self, save_path: str = None):
        """可视化实验结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 类内离散度对比 (柱状图)
        ax1 = axes[0, 0]
        data_for_plot = []
        for dataset in ['MFD', 'C-MFD']:
            for category in self.categories:
                for polarity in self.polarities:
                    try:
                        cos_sim = self.results['intra_class_dispersion'][dataset][category][polarity]['avg_cosine_sim']
                        data_for_plot.append({
                            'Dataset': dataset,
                            'Category': category,
                            'Polarity': polarity,
                            'Avg Cosine Similarity': cos_sim
                        })
                    except KeyError:
                        continue

        plot_df = pd.DataFrame(data_for_plot)
        if not plot_df.empty:
            pivot_df = plot_df.pivot_table(
                values='Avg Cosine Similarity',
                index=['Dataset', 'Category'],
                columns='Polarity'
            )
            pivot_df.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('类内平均余弦相似度 (Virtue vs Vice)')
            ax1.set_ylabel('Average Cosine Similarity')
            ax1.legend(title='Polarity')
            ax1.tick_params(axis='x', rotation=45)

        # 2. Virtue-Vice方向相似度
        ax2 = axes[0, 1]
        direction_data = []
        for dataset in ['MFD', 'C-MFD']:
            for category in self.categories:
                try:
                    cos_sim = self.results['virtue_vice_direction'][dataset][category]['cosine_similarity']
                    direction_data.append({
                        'Dataset': dataset,
                        'Category': category,
                        'Cosine Similarity': cos_sim
                    })
                except KeyError:
                    continue

        direction_df = pd.DataFrame(direction_data)
        if not direction_df.empty:
            pivot_direction = direction_df.pivot(index='Category', columns='Dataset', values='Cosine Similarity')
            pivot_direction.plot(kind='bar', ax=ax2)
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax2.axhline(y=-0.5, color='red', linestyle=':', linewidth=0.5, label='Opposite threshold')
            ax2.set_title('Virtue与Vice中心向量的余弦相似度')
            ax2.set_ylabel('Cosine Similarity')
            ax2.tick_params(axis='x', rotation=45)

        # 3. 跨数据集相似度
        ax3 = axes[1, 0]
        cross_data = []
        for category in self.categories:
            for polarity in self.polarities:
                try:
                    cos_sim = self.results['cross_dataset_similarity'][category][polarity]['centroid_cosine_similarity']
                    cross_data.append({
                        'Category': category,
                        'Polarity': polarity,
                        'Cosine Similarity': cos_sim
                    })
                except KeyError:
                    continue

        cross_df = pd.DataFrame(cross_data)
        if not cross_df.empty:
            pivot_cross = cross_df.pivot(index='Category', columns='Polarity', values='Cosine Similarity')
            pivot_cross.plot(kind='bar', ax=ax3)
            ax3.set_title('跨数据集 (MFD vs C-MFD) 类别中心相似度')
            ax3.set_ylabel('Cosine Similarity')
            ax3.tick_params(axis='x', rotation=45)

        # 4. 假设验证热力图
        ax4 = axes[1, 1]
        if 'virtue_vice_comparison' in self.results:
            comparison_df = self.results['virtue_vice_comparison']
            heatmap_data = comparison_df.pivot(index='category', columns='dataset', values='difference')
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('Virtue-Vice离散度差异\n(正值=Virtue更集中,支持假设)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图表已保存至: {save_path}")

        plt.show()
        return fig

    def save_results(self, output_dir: str):
        """保存实验结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存离散度分析结果
        dispersion_records = []
        for dataset in ['MFD', 'C-MFD']:
            for category in self.categories:
                for polarity in self.polarities:
                    try:
                        disp = self.results['intra_class_dispersion'][dataset][category][polarity]
                        record = {
                            'dataset': dataset,
                            'category': category,
                            'polarity': polarity,
                            **disp
                        }
                        dispersion_records.append(record)
                    except KeyError:
                        continue

        pd.DataFrame(dispersion_records).to_csv(
            os.path.join(output_dir, 'intra_class_dispersion.csv'), index=False)

        # 保存方向分析结果
        direction_records = []
        for dataset in ['MFD', 'C-MFD']:
            for category in self.categories:
                try:
                    rel = self.results['virtue_vice_direction'][dataset][category]
                    record = {'dataset': dataset, 'category': category, **rel}
                    direction_records.append(record)
                except KeyError:
                    continue

        pd.DataFrame(direction_records).to_csv(
            os.path.join(output_dir, 'virtue_vice_direction.csv'), index=False)

        # 保存跨数据集分析结果
        cross_records = []
        for category in self.categories:
            for polarity in self.polarities:
                try:
                    sim = self.results['cross_dataset_similarity'][category][polarity]
                    record = {'category': category, 'polarity': polarity, **sim}
                    cross_records.append(record)
                except KeyError:
                    continue

        pd.DataFrame(cross_records).to_csv(
            os.path.join(output_dir, 'cross_dataset_similarity.csv'), index=False)

        # 保存对比分析结果
        if 'virtue_vice_comparison' in self.results:
            self.results['virtue_vice_comparison'].to_csv(
                os.path.join(output_dir, 'virtue_vice_comparison.csv'), index=False)

        print(f"\n结果已保存至: {output_dir}")


# =============================================================================
# 6. 主函数
# =============================================================================

def main():
    """主函数"""
    # 数据路径
    MFD_PATH = "./datasets/MFD.csv"
    CMFD_PATH = "./datasets/C-MFD.csv"
    OUTPUT_DIR = "./results/moral_experiment_results"

    # 创建实验对象
    experiment = MoralRepresentationExperiment(
        mfd_path=MFD_PATH,
        cmfd_path=CMFD_PATH,
        model_name=os.environ.get("MODEL_PATH", "meta-llama/Llama-3.1-8B")
    )

    # 运行完整实验
    results = experiment.run_full_experiment(use_whitening=True)

    # 可视化结果
    experiment.visualize_results(save_path=os.path.join(OUTPUT_DIR, 'results_visualization.png'))

    # 保存结果
    experiment.save_results(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)

    return experiment, results


if __name__ == "__main__":
    main()