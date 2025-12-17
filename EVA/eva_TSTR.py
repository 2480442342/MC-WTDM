import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

import os
from datetime import datetime
import re

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

now = datetime.now().strftime("%Y-%m-%d-%H")

# ==========================================
# 1. MLP 判别器 (针对 2D 表格数据)
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
    计算判别分数
    real_data: (N, Dim) 2D Array
    fake_data: (M, Dim) 2D Array
    """
    # 转换为 Tensor
    real_tensor = torch.from_numpy(real_data).float().to(device)
    fake_tensor = torch.from_numpy(fake_data).float().to(device)
    
    # 构造标签: Real=1, Fake=0
    labels_real = torch.ones(len(real_tensor), 1).to(device)
    labels_fake = torch.zeros(len(fake_tensor), 1).to(device)
    
    # 合并数据
    data = torch.cat([real_tensor, fake_tensor], dim=0)
    labels = torch.cat([labels_real, labels_fake], dim=0)
    
    # 初始化 MLP 判别器
    disc = MLPDiscriminator(input_dim=real_tensor.shape[1]).to(device)
    optimizer = torch.optim.Adam(disc.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Training MLP Discriminator for {epochs} epochs...")
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            pred = disc(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
    
    # 计算准确率 (越接近 0.5 越好)
    with torch.no_grad():
        preds = disc(data).cpu().numpy() > 0.5
        y_true = labels.cpu().numpy()
        acc = accuracy_score(y_true, preds)
        
    print(f"Discriminator Accuracy: {acc:.4f}")
    return abs(0.5 - acc) # Discriminative Score

# ==========================================
# 【新增】 2. 统计距离评估 (FID & Wasserstein)
# ==========================================
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical instability might lead to small complex numbers
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Only take real component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def get_statistics(data, scaler=None):
    """
    计算均值和协方差，并支持标准化
    :param data: (N, D) 形状的数据
    :param scaler: 用于标准化的 scaler (mean, std)
    """
    # 1. 维度检查与展平：如果是 (N, Seq, Feat)，展平为 (N, Seq*Feat)
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    
    # 2. 【核心优化】标准化 (Z-Score Normalization)
    # 必须使用真实数据的统计量来标准化生成数据，保证相对距离的意义
    if scaler is None:
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0) + 1e-6 # 防止除零
        scaler = (mean_val, std_val)
    
    mean_ref, std_ref = scaler
    # 执行标准化：将数据映射到 N(0, 1) 附近，大幅降低数值量级
    data_norm = (data - mean_ref) / std_ref
    
    mu = np.mean(data_norm, axis=0)
    sigma = np.cov(data_norm, rowvar=False)
    
    return mu, sigma, scaler

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, normalize_by_dim=False):
    """
    Numpy implementation of the Frechet Distance with Optimization.
    优化点：增加了数值稳定性和可选的维度归一化
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical instability might lead to small complex numbers
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        # print(msg) # 可以注释掉减少日志
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Only take real component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    # 原始 Squared Frechet Distance
    fid_squared = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    # 【优化点】防止计算误差导致的微小负数
    if fid_squared < 0:
        fid_squared = 0
        
    # 【优化点】返回开方后的值 (Distance 而不是 Squared Distance)
    # 这样量级会从 18000 -> sqrt(18000) ≈ 134
    fid = np.sqrt(fid_squared)
    
    # 【可选优化】如果维度特别高，除以维度数进行平均
    if normalize_by_dim:
        fid = fid / mu1.shape[0]

    return fid

# ==========================================
# 调用示例 (封装流程)
# ==========================================
def evaluate_fid_score(real_data, pred_data):
    """
    封装后的评估函数
    real_data: (Batch, Seq_Len, Dim)
    pred_data: (Batch, Seq_Len, Dim)
    """
    # 1. 转为 Numpy
    if hasattr(real_data, 'cpu'): real_data = real_data.cpu().numpy()
    if hasattr(pred_data, 'cpu'): pred_data = pred_data.cpu().numpy()
    
    # 2. 计算真实数据的统计量，并获取标准化参数
    mu1, sigma1, scaler = get_statistics(real_data, scaler=None)
    
    # 3. 使用真实数据的参数 标准化 预测数据 (关键步骤)
    mu2, sigma2, _ = get_statistics(pred_data, scaler=scaler)
    
    # 4. 计算距离
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid_score

def evaluate_statistical_metrics(real_data, syn_data, feature_names=None):
    """
    计算 Wasserstein 距离 和 Fréchet Distance
    优化：对 WD 也进行标准化，消除物理量级的影响
    """
    print("-" * 30)
    print("Evaluating Statistical Metrics...")
    
    # 维度检查与展平
    if real_data.ndim > 2:
        real_data = real_data.reshape(real_data.shape[0], -1)
    if syn_data.ndim > 2:
        syn_data = syn_data.reshape(syn_data.shape[0], -1)

    # 1. Wasserstein Distance (Normalized)
    wd_list = []
    N, D = real_data.shape
    
    # 【关键修改】预计算真实数据的统计量，用于标准化
    # 这样比较的是相对分布形态，而不是绝对数值差异
    real_mean = np.mean(real_data, axis=0)
    real_std = np.std(real_data, axis=0) + 1e-6  # 防止除以0
    
    for i in range(D):
        # 使用真实数据的均值和方差对 两者 都进行标准化
        # 这样可以看出生成分布相对于真实分布的偏差（以标准差为单位）
        r_norm = (real_data[:, i] - real_mean[i]) / real_std[i]
        s_norm = (syn_data[:, i] - real_mean[i]) / real_std[i]
        
        wd = wasserstein_distance(r_norm, s_norm)
        wd_list.append(wd)
    
    avg_wd = np.mean(wd_list)
    
    # 2. Fréchet Distance (FD) - (假设该函数内部已包含标准化)
    fd_score = evaluate_fid_score(real_data, syn_data)
    
    # --- 打印结果 ---
    print(f"1. Avg Wasserstein Dist (Normalized): {avg_wd:.4f} (Lower is better)")
    # 现在的 WD 通常应该在 0.0x 到 1.0 之间。如果 > 1.0 说明分布差异非常大。
    
    print(f"2. Fréchet Distance (FD):             {fd_score:.4f} (Lower is better)")
    
    # 打印差异最大的特征（基于标准化后的 WD，更能反映真实的分布偏离）
    if feature_names:
        # 确保 feature_names 长度匹配
        if len(feature_names) != D:
             # 如果数据被展平了（Sequence * Features），这里可能需要调整名字
             print("   (Feature names alignment skipped due to flattening)")
        else:
            sorted_indices = np.argsort(wd_list)[::-1]
            print("   Top 3 features with largest Distribution Shift (Normalized WD):")
            for idx in sorted_indices[:3]:
                fname = feature_names[idx]
                print(f"   - {fname}: {wd_list[idx]:.4f}")

    print("-" * 30)
    
    return avg_wd, fd_score

# ==========================================
# 2. 可视化分析 (适配 2D 数据)
# ==========================================
def evaluate_multivariate_data(real_data, syn_data, feature_names=None, string_date=None):
    """
    real_data: (N, D)
    syn_data: (M, D)
    """
    N, D = real_data.shape
    
    if feature_names is None:
        feature_names = [f'Feat_{i}' for i in range(D)]

    # --- 1. 相关性矩阵对比 ---
    # 直接计算 (Feature, Feature) 相关性
    corr_real = np.corrcoef(real_data.T)
    corr_syn = np.corrcoef(syn_data.T)
    
    # 处理可能的无效值
    corr_real = np.nan_to_num(corr_real)
    corr_syn = np.nan_to_num(corr_syn)
    
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    sns.heatmap(corr_real, ax=ax[0], cmap='coolwarm', vmin=-1, vmax=1)
    ax[0].set_title("Real Data Correlation")
    
    sns.heatmap(corr_syn, ax=ax[1], cmap='coolwarm', vmin=-1, vmax=1)
    ax[1].set_title("Synthetic Data Correlation")
    
    diff = np.abs(corr_real - corr_syn)
    sns.heatmap(diff, ax=ax[2], cmap='Reds', vmin=0, vmax=0.5)
    ax[2].set_title(f"Difference (Frobenius: {np.linalg.norm(diff):.4f})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{string_date}_Correlation_Comparison.tif'), dpi=300)
    # plt.show()
    
    # --- 2. 逐特征分布对比 (KDE) ---
    # 随机选 4 个特征展示
    selected_feats = np.random.choice(range(D), min(D, 4), replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    # 为了绘图速度，采样最多 5000 个点
    idx_r = np.random.choice(len(real_data), min(len(real_data), 5000), replace=False)
    idx_s = np.random.choice(len(syn_data), min(len(syn_data), 5000), replace=False)
    
    for i, feat_idx in enumerate(selected_feats):
        sns.kdeplot(real_data[idx_r, feat_idx], ax=axes[i], color='blue', label='Real', fill=True, alpha=0.1)
        sns.kdeplot(syn_data[idx_s, feat_idx], ax=axes[i], color='red', label='Syn', linestyle='--')
        
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else str(feat_idx)
        axes[i].set_title(f"{fname}")
        if i == 0: axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{string_date}_KDE_Comparison.tif'), dpi=300)
    # plt.show()

    # --- 3. t-SNE 可视化 ---
    print("Calculating t-SNE...")
    # 采样 1000 个点进行 t-SNE (点太多会很慢且看不清)
    num_samples = min(len(real_data), len(syn_data), 1000)
    
    idx_real = np.random.choice(len(real_data), num_samples, replace=False)
    idx_syn = np.random.choice(len(syn_data), num_samples, replace=False)
    
    X_combined = np.concatenate([real_data[idx_real], syn_data[idx_syn]], axis=0)
    
    # 【修改点1】：直接生成字符串标签，而不是 0 和 1
    # 这样 Seaborn 就能直接识别并显示正确的图例文本
    labels = np.concatenate([['Real'] * num_samples, ['Synthetic'] * num_samples])
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X_combined)
    
    plt.figure(figsize=(8, 6))
    
    # 【修改点2】：palette 的键改为对应的字符串，hue 传入字符串标签
    sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=labels, 
                    palette={'Real':'blue', 'Synthetic':'red'}, 
                    alpha=0.6, s=20)
    
    # 【修改点3】：移除 labels 参数，只设置标题
    # Seaborn 已经根据 hue 自动生成了正确的颜色和标签，这里只需要设置图例标题
    plt.legend(title='Data Type')
    
    plt.title("t-SNE Comparison (2D Data)")
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{string_date}_TSNE_2D.tif'), dpi=300)
    # plt.show()

# ==========================================
# 3. 数据加载 (适配 2D 形状)
# ==========================================
def load_2d_data(csv_path, npz_path, col_names=None):
    """
    加载并对齐数据形状为 (N, D)
    """
    # 1. 加载生成数据
    if npz_path.endswith('.npz'):
        syn_pack = np.load(npz_path)
        # 假设 'data' 里的数据。注意：如果原来是 (N, C, 1) 或者 (N, 1, C)，需要压缩维度
        if 'DM_MTS' in npz_path or npz_path.endswith('generate_dataset.npz'):
            syn_data = syn_pack['data'][:, np.random.randint(0, syn_pack['data'].shape[1]), :]
        elif npz_path.endswith('WT_feat21.npz') or npz_path.endswith('WT_feat19.npz') or npz_path.endswith('WT_feat19v2.npz') or npz_path.endswith('WT_feat16.npz'):
            syn_data = syn_pack['data'][:, :, np.random.randint(0, syn_pack['data'].shape[2])]
        elif npz_path.endswith('.npz') and not npz_path.endswith('generated_result.npz'):
            syn_data = syn_pack['data'][:, 4:]
            num_cols = syn_data.shape[1]
            mask = np.ones(num_cols, dtype=bool)
            mask[12:14] = False 
            syn_data = syn_data[:, mask]
        elif npz_path.endswith('generated_result.npz'):
            syn_data = syn_pack['data'][:, 7:]
        else:
            syn_data = syn_pack[:, 4:]
    elif npz_path.endswith('2025-05-22-20_N2_WindTurbine.csv'):
        # 1. 读取 CSV 并直接转换为 Numpy 数组 (float32)
        # 注意：这里先把 df 变成 numpy array，后续逻辑才通顺
        raw_data = pd.read_csv(npz_path, header=0).values.astype(np.float32)
        
        # 2. 修改变量名：使用 raw_data 而不是 syn_pack
        syn_data = raw_data[:, 8:]  # 去掉前4列
        
        # 3. 后续逻辑保持不变
        num_cols = syn_data.shape[1]
        mask = np.ones(num_cols, dtype=bool)
        mask[13:18] = False  # 去掉切片后的第12、13列
    
        # 应用掩码
        syn_data = syn_data[:, mask]
    
    # 挤压维度：(N, C, 1) -> (N, C)
    if syn_data.ndim == 3:
        # 如果是 (N, 21, 1)，去掉最后一维
        if syn_data.shape[2] == 1:
            syn_data = syn_data.squeeze(2)
        # 如果是 (N, 1, 21)，去掉中间维
        elif syn_data.shape[1] == 1:
            syn_data = syn_data.squeeze(1)
            
    print(f"Synthetic Data Shape: {syn_data.shape}") # 期望是 (N, 21)
    
    # 2. 加载真实数据
    if csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path, header=0)
        if col_names:
            # 确保列顺序一致，且只取特征列
            # real_data = df[col_names].values.astype(np.float32)
            real_data = df[col_names].sample(n=len(syn_data), replace=False, random_state=42).values.astype(np.float32)
        else:
            real_data = df.values.sample(n=len(syn_data), replace=False, random_state=42).astype(np.float32)
    elif csv_path.endswith('.npz'):
        real_npz = np.load(csv_path)
        real_data = real_npz['data'][:, np.random.randint(0, real_npz['data'].shape[1]), :]

    print(f"Real Data Shape: {real_data.shape}") # 期望是 (M, 16)
    
    # 维度一致性检查
    if syn_data.shape[1] != real_data.shape[1]:
        raise ValueError(f"Feature dimension mismatch! Real: {real_data.shape[1]}, Syn: {syn_data.shape[1]}")
        
    return real_data, syn_data

if __name__ == '__main__':
    # 替换为你的实际路径
    real_path = r'E:\CMPASS\AMyProject\data\dap_tenmindata_7_202101_cleaned.csv'
    syn_path = r'E:\CMPASS\RTX3080ti\weights\syn_data_test\syn_2025-12-17-09_WT_feat16.npz'
    # 1. 获取基础文件名和保存目录
    file_name = os.path.basename(syn_path)
    save_dir = os.path.dirname(__file__)

    # 2. 提取核心标识符 (增加异常处理防止正则匹配失败)
    match = re.search(r'syn_(.*?)_WT_feat16', file_name)
    if match:
        base_string = match.group(1)
    else:
        # 如果正则没匹配到，兜底使用文件名去掉后缀
        base_string = os.path.splitext(file_name)[0]

    # 3. 自动递增标号逻辑
    final_string = base_string
    counter = 1

    # 检查文件是否存在，如果存在则循环尝试 v1, v2, v3...
    # 这里以 '_Correlation_Comparison.png' 为检测锚点
    while os.path.exists(os.path.join(save_dir, f'{final_string}_Correlation_Comparison.tif')):
        print(f"File with ID '{final_string}' exists. Incrementing version...")
        final_string = f"{base_string}_v{counter}"
        counter += 1

    # 4. 最终结果
    print(f"Final unique ID to use: {final_string}")

    csv_columns = [
        'MainBearingSpeedMean', 'GeneratorSpeedMean', 'MainBearingTempFrontMean', 'MainBearingTempBackMean', 
        'GearboxDEBearingTempMean', 'GearboxNDEBearingTempMean', 'GearboxOilSumpTempMean', 'GeneratorDEBearingTempMean', 
        'GeneratorNDEBearingTempMean', 'GeneratorWindingTempUMean', 'GeneratorWindingTempVMean', 'GeneratorWindingTempWMean', 
        'YawErrorMean', 'GridPhaseCurrentABMean', 'GridPhaseCurrentBCMean', 'GridPhaseCurrentCAMean'
    ]
    
    try:
        # 1. 加载数据
        real_data, syn_data = load_2d_data(real_path, syn_path, csv_columns)
        
        # 2. 判别分数
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        score = evaluate_discriminative_score(real_data, syn_data, epochs=20, device=device)
        print(f"Discriminative Score: {score:.4f} (Lower is better)")

        # 3. 【新增】统计距离 (Wasserstein & Fréchet)
        wd_score, fd_score = evaluate_statistical_metrics(real_data, syn_data, csv_columns)

        print("="*40)
        print("Summary of Evaluation Metrics:")
        print(f"1. Discriminative Score: {score:.4f} (Ideal: 0.0)")
        print(f"2. Avg Wasserstein Dist: {wd_score:.4f} (Ideal: 0.0)")
        print(f"3. Fréchet Distance  : {fd_score:.4f} (Ideal: 0.0)")
        print("="*40)
        
        # 3. 可视化
        evaluate_multivariate_data(real_data, syn_data, csv_columns, string_date=final_string)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        