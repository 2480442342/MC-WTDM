import os
import re
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import random # 记得导入 random

# 1. 全局绘图配置 (关键修复部分)
# ==========================================
# 【重要修复】必须先设置 Seaborn 风格，否则它会覆盖后面的字体设置
sns.set_style("whitegrid")

# 在 Seaborn 设置之后，强制覆盖字体配置
# 设置公式字体为 stix (类似 Times New Roman 的数学字体)
plt.rcParams['mathtext.fontset'] = 'stix'
# 设置全局字体族为衬线体 (serif)
plt.rcParams['font.family'] = 'serif'
# 注意：列表顺序决定查找优先权
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 颜色盘：Real=蓝色, Synthetic=红色
COLOR_PALETTE = {'Real': '#1f77b4', 'Synthetic': '#d62728'}
# 中文图例映射
LABEL_MAP = {'Real': '真实数据', 'Synthetic': '生成数据'}

def seed_everything(seed=42):
    """固定所有随机种子，确保结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 全局随机种子已固定为: {seed}")
# ==========================================

def save_figure(save_path):
    """
    通用保存图片逻辑 (论文专用版)
    1. DPI >= 300
    2. 无留白 (pad_inches=0)
    """
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            pass
    try:
        # 使用 png 格式兼容性更好，dpi=300 保证高清
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        print(f"图表已保存到: {save_path}")
    except Exception as e:
        print(f"保存失败: {e}")


# ==========================================
# 2. MLP 判别器 (Discriminative Score)
# ==========================================
class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def evaluate_discriminative_score(real_data, fake_data, epochs=20, device='cpu'):
    """
    训练一个分类器区分真假数据。
    结果越接近 0 (即 accuracy 接近 0.5)，说明生成越逼真。
    """
    real_tensor = torch.from_numpy(real_data).float().to(device)
    fake_tensor = torch.from_numpy(fake_data).float().to(device)

    labels_real = torch.ones(len(real_tensor), 1).to(device)
    labels_fake = torch.zeros(len(fake_tensor), 1).to(device)

    data = torch.cat([real_tensor, fake_tensor], dim=0)
    labels = torch.cat([labels_real, labels_fake], dim=0)

    disc = MLPDiscriminator(input_dim=real_tensor.shape[1]).to(device)
    optimizer = torch.optim.Adam(disc.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"正在训练 MLP 判别器 ({epochs} epochs)...")
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            pred = disc(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = disc(data).cpu().numpy() > 0.5
        y_true = labels.cpu().numpy()
        acc = accuracy_score(y_true, preds)

    print(f"判别器准确率: {acc:.4f}")
    return abs(0.5 - acc)  # Discriminative Score


# ==========================================
# 3. 统计距离评估 (FID & Wasserstein)
# ==========================================
def get_statistics(data, scaler=None):
    """计算均值和协方差，并支持标准化"""
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if scaler is None:
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0) + 1e-6
        scaler = (mean_val, std_val)

    mean_ref, std_ref = scaler
    data_norm = (data - mean_ref) / std_ref

    mu = np.mean(data_norm, axis=0)
    sigma = np.cov(data_norm, rowvar=False)

    return mu, sigma, scaler


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算 Fréchet Distance (FD)"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid_squared = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    if fid_squared < 0:
        fid_squared = 0
    return np.sqrt(fid_squared)


def evaluate_statistical_metrics(real_data, syn_data, feature_names=None):
    """计算 Wasserstein 距离 和 Fréchet Distance"""
    print("-" * 30)
    print("正在评估统计指标...")

    # 展平数据
    if real_data.ndim > 2:
        real_data = real_data.reshape(real_data.shape[0], -1)
    if syn_data.ndim > 2:
        syn_data = syn_data.reshape(syn_data.shape[0], -1)

    # 1. Wasserstein Distance (Normalized)
    wd_list = []
    N, D = real_data.shape
    real_mean = np.mean(real_data, axis=0)
    real_std = np.std(real_data, axis=0) + 1e-6

    for i in range(D):
        r_norm = (real_data[:, i] - real_mean[i]) / real_std[i]
        s_norm = (syn_data[:, i] - real_mean[i]) / real_std[i]
        wd = wasserstein_distance(r_norm, s_norm)
        wd_list.append(wd)

    avg_wd = np.mean(wd_list)

    # 2. Fréchet Distance (FD)
    mu1, sigma1, scaler = get_statistics(real_data, scaler=None)
    mu2, sigma2, _ = get_statistics(syn_data, scaler=scaler)
    fd_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    print(f"1. 平均 Wasserstein 距离 (Normalized): {avg_wd:.4f}")
    print(f"2. Fréchet 距离 (FD):                {fd_score:.4f}")

    if feature_names and len(feature_names) == D:
        sorted_indices = np.argsort(wd_list)[::-1]
        print("   分布偏移最大的前3个特征:")
        for idx in sorted_indices[:3]:
            print(f"   - {feature_names[idx]}: {wd_list[idx]:.4f}")

    print("-" * 30)
    return avg_wd, fd_score


# ==========================================
# 4. 可视化分析 (核心修改部分)
# ==========================================
def evaluate_multivariate_data(real_data, syn_data, feature_names=None, save_prefix=None):
    """
    可视化评估：相关性矩阵、KDE、PCA 和 t-SNE
    已适配论文格式：无标题、高清、Times+宋体
    """
    if save_prefix is None:
        save_prefix = datetime.now().strftime("%Y-%m-%d-%H")

    N, D = real_data.shape
    if feature_names is None:
        feature_names = [f'Feat_{i}' for i in range(D)]

    # --- 1. 相关性矩阵 (Correlation Matrix) ---
    corr_real = np.nan_to_num(np.corrcoef(real_data.T))
    corr_syn = np.nan_to_num(np.corrcoef(syn_data.T))

    # 设置画布，无标题模式
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # 真实数据热力图
    sns.heatmap(corr_real, ax=ax[0], cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
    ax[0].set_xlabel('真实数据相关性', fontsize=14)

    # 生成数据热力图
    sns.heatmap(corr_syn, ax=ax[1], cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
    ax[1].set_xlabel('生成数据相关性', fontsize=14)

    # 差异热力图
    diff = np.abs(corr_real - corr_syn)
    sns.heatmap(diff, ax=ax[2], cmap='Reds', vmin=0, vmax=0.5)
    ax[2].set_xlabel(f'差异绝对值 (范数: {np.linalg.norm(diff):.2f})', fontsize=14)

    plt.tight_layout()
    save_figure(f'{save_prefix}_Correlation.png')
    plt.close()

    # --- 2. KDE 分布图 (采样 4 个特征) ---
    # 随机选择 4 个特征进行展示
    selected_feats = np.random.choice(range(D), min(D, 4), replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # 为了画图速度，进行降采样
    idx_r = np.random.choice(len(real_data), min(len(real_data), 5000), replace=False)
    idx_s = np.random.choice(len(syn_data), min(len(syn_data), 5000), replace=False)

    for i, feat_idx in enumerate(selected_feats):
        # 真实数据
        sns.kdeplot(
            real_data[idx_r, feat_idx],
            ax=axes[i],
            color=COLOR_PALETTE['Real'],
            label=LABEL_MAP['Real'],
            fill=True,
            alpha=0.1,
            linewidth=2
        )
        # 生成数据
        sns.kdeplot(
            syn_data[idx_s, feat_idx],
            ax=axes[i],
            color=COLOR_PALETTE['Synthetic'],
            label=LABEL_MAP['Synthetic'],
            linestyle='--',
            linewidth=2
        )

        # 获取特征名
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else str(feat_idx)

        # 设置标签
        axes[i].set_xlabel(fname, fontsize=12)
        axes[i].set_ylabel('概率密度', fontsize=12)

        # 仅在第一个子图显示图例
        if i == 0:
            axes[i].legend(fontsize=10, frameon=True)

    plt.tight_layout()
    save_figure(f'{save_prefix}_KDE.png')
    plt.close()

    # --- 3. 流形学习可视化 (PCA & t-SNE) ---
    print("正在计算 PCA 和 t-SNE...")
    num_samples = min(len(real_data), len(syn_data), 1000)  # 采样以加速

    idx_real = np.random.choice(len(real_data), num_samples, replace=False)
    idx_syn = np.random.choice(len(syn_data), num_samples, replace=False)

    X_combined = np.concatenate([real_data[idx_real], syn_data[idx_syn]], axis=0)
    labels = np.concatenate([[LABEL_MAP['Real']] * num_samples, [LABEL_MAP['Synthetic']] * num_samples])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A. PCA (全局结构)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)

    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=labels,
        palette={LABEL_MAP['Real']: COLOR_PALETTE['Real'], LABEL_MAP['Synthetic']: COLOR_PALETTE['Synthetic']},
        alpha=0.6, s=30, ax=axes[0], edgecolor='w', linewidth=0.3
    )
    axes[0].set_xlabel("主成分 1", fontsize=12)
    axes[0].set_ylabel("主成分 2", fontsize=12)
    axes[0].legend(fontsize=10)

    # B. t-SNE (局部结构)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_combined)

    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=labels,
        palette={LABEL_MAP['Real']: COLOR_PALETTE['Real'], LABEL_MAP['Synthetic']: COLOR_PALETTE['Synthetic']},
        alpha=0.6, s=30, ax=axes[1], edgecolor='w', linewidth=0.3
    )
    axes[1].set_xlabel("维度 1", fontsize=12)
    axes[1].set_ylabel("维度 2", fontsize=12)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save_figure(f'{save_prefix}_Manifold_Comparison.png')
    plt.close()
    print(f"所有可视化图表已保存，前缀: {save_prefix}")


# ==========================================
# 5. 数据加载辅助函数
# ==========================================
def load_2d_data(real_path, syn_path, col_names=None):
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real data not found: {real_path}")
    if not os.path.exists(syn_path):
        raise FileNotFoundError(f"Syn data not found: {syn_path}")

    # 1. 加载生成数据
    syn_data = None
    if syn_path.endswith('.npz'):
        syn_pack = np.load(syn_path)
        keys = list(syn_pack.keys())
        data_key = 'data' if 'data' in keys else keys[0]
        raw_syn = syn_pack[data_key]

        if raw_syn.ndim == 3:
            # (N, Seq, Feat) -> 取随机时间步
            syn_data = raw_syn[:, np.random.randint(0, raw_syn.shape[1]), :] if raw_syn.shape[1] > 1 else raw_syn.squeeze(1)
            if raw_syn.shape[2] == 1:
                syn_data = raw_syn.squeeze(2)
        else:
            syn_data = raw_syn

        if 'WT_feat' in syn_path and syn_data.shape[1] > 16:
            syn_data = syn_data[:, :16]

    elif syn_path.endswith('.csv'):
        syn_data = pd.read_csv(syn_path).values.astype(np.float32)

    # 2. 加载真实数据
    if real_path.endswith('.csv'):
        df = pd.read_csv(real_path)
        if col_names:
            available_cols = [c for c in col_names if c in df.columns]
            real_data = df[available_cols].values
        else:
            real_data = df.values
    else:
        real_data = np.load(real_path)['data']

    # 3. 维度对齐
    min_cols = min(real_data.shape[1], syn_data.shape[1])
    real_data = real_data[:, :min_cols]
    syn_data = syn_data[:, :min_cols]

    print(f"数据加载完毕. Real: {real_data.shape}, Syn: {syn_data.shape}")
    return real_data.astype(np.float32), syn_data.astype(np.float32)


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # random seed
    seed_everything(15)

    # 配置路径
    base_dir = r'E:\CMPASS\AMyProject'
    real_path = os.path.join(base_dir, 'data', 'dap_tenmindata_7_id_46_test.csv')
    # syn_path = os.path.join(base_dir, 'weights', 'syn_data_test', 'syn_2025-12-13-11_WT_feat16.npz')
    syn_path = r"E:\CMPASS\AGitHub\MC-CMSDM-Compared\weights\syn_data_test\syn_2026-03-18-16_WT_feat17.npz"

    # 生成文件前缀
    file_name = os.path.basename(syn_path)
    base_name = os.path.splitext(file_name)[0]
    save_path_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_name)

    # 定义列名 (如果需要汉化，建议在此处建立映射字典，然后在 evaluate_multivariate_data 中替换)
    csv_columns = [
        'MainBearingSpeedMean', 'GeneratorSpeedMean', 'MainBearingTempFrontMean', 'MainBearingTempBackMean',
        'GearboxDEBearingTempMean', 'GearboxNDEBearingTempMean', 'GearboxOilSumpTempMean', 'GeneratorDEBearingTempMean',
        'GeneratorNDEBearingTempMean', 'GeneratorWindingTempUMean', 'GeneratorWindingTempVMean', 'GeneratorWindingTempWMean',
        'YawErrorMean', 'GridPhaseCurrentABMean', 'GridPhaseCurrentBCMean', 'GridPhaseCurrentCAMean'
    ]

    try:
        # 1. 加载数据
        print(">>> 正在加载数据...")
        real_data, syn_data = load_2d_data(real_path, syn_path, csv_columns)

        # 2. 判别性指标
        print("\n>>> 1. 计算判别性分数 (Discriminative Score)")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        disc_score = evaluate_discriminative_score(real_data, syn_data, epochs=20, device=device)

        # 3. 统计性指标
        print("\n>>> 2. 计算统计距离 (Statistical Metrics)")
        wd_score, fd_score = evaluate_statistical_metrics(real_data, syn_data, csv_columns)

        # 4. 可视化
        print("\n>>> 3. 生成可视化图表 (t-SNE & Correlation)")
        evaluate_multivariate_data(real_data, syn_data, csv_columns, save_prefix=save_path_prefix)

        print("\n" + "=" * 40)
        print("最终评估报告")
        print(f"文件名: {base_name}")
        print(f"判别性分数 (与0.5的差距): {disc_score:.4f}")
        print(f"Fréchet 距离 (FD):       {fd_score:.4f}")
        print(f"平均 Wasserstein 距离:   {wd_score:.4f}")
        print("=" * 40)

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        traceback.print_exc()