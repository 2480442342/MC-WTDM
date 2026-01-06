import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. 基础组件 (AdaLN & Embeddings)
# ==============================================================================

def modulate(x, shift, scale):
    """
    AdaLN 核心操作: 对归一化后的特征进行平移(shift)和缩放(scale)
    x: (B, L, D)
    shift, scale: (B, D) -> (B, 1, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    将标量时间步 t 映射为向量嵌入
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        生成正弦位置编码
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

# ==============================================================================
# 2. 核心模块 (DiT Block)
# ==============================================================================

class DiTBlock(nn.Module):
    """
    基于 Transformer 的核心块，使用 AdaLN 注入条件信息。
    支持不同的 hidden_size (当前层维度) 和 cond_dim (全局条件维度)。
    """
    def __init__(self, hidden_size, num_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 自动调整 num_heads，防止 hidden_size 过小时报错
        # 确保 head_dim 至少为 4
        if hidden_size % num_heads != 0:
            num_heads = 4 
            
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # AdaLN Modulation: 
        # 输入: 全局条件 c (cond_dim)
        # 输出: 针对当前层的 6 个参数 (shift/scale/gate for MSA & MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

        # Zero-init: 初始化为 0，使得初始训练像恒等映射，加速收敛
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # x: (Batch, Seq_Len, Hidden_Size)
        # c: (Batch, Cond_Dim)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # 1. Attention Block
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. MLP Block
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x

class FinalLayer(nn.Module):
    """DiT 风格的最终输出层"""
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# ==============================================================================
# 3. 上下采样 (适配 Transformer 格式)
# ==============================================================================

class Downsample(nn.Module):
    """使用卷积下采样: (B, L, C) -> (B, L/2, C_out)"""
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        # Kernel=3, Stride=2 减少长度
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # Transpose for Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class Upsample(nn.Module):
    """使用插值+卷积上采样: (B, L, C) -> (B, L*2, C_out)"""
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.up(x)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

# ==============================================================================
# 4. 主网络架构 (U-DiT)
# ==============================================================================

class U_DiT(nn.Module):
    def __init__(
        self,
        dim=32,                
        dim_mults=(1, 2, 4),   
        channels=1,            
        feature_dim=None,      
        cond_drop_prob=0.1,    
        num_classes=4,         
        **kwargs
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        
        input_channels = channels 
        condition_dim = num_classes 
        
        # 1. 嵌入层
        self.x_embedder = nn.Linear(input_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = nn.Linear(condition_dim, dim)
        
        # 2. 构建 U-Net 骨架
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        dims = [dim, *map(lambda m: dim * m, dim_mults)] 
        in_out = list(zip(dims[:-1], dims[1:]))          
        
        c_dim = dim 
        
        # --- Encoder (Down Path) ---
        for dim_in, dim_out in in_out:
            self.downs.append(nn.ModuleList([
                DiTBlock(dim_in, num_heads=4, cond_dim=c_dim), 
                DiTBlock(dim_in, num_heads=4, cond_dim=c_dim),
                Downsample(dim_in, dim_out)
            ]))
            
        mid_dim = dims[-1]
        
        # --- Middle Path ---
        self.mid_block1 = DiTBlock(mid_dim, num_heads=8, cond_dim=c_dim)
        self.mid_block2 = DiTBlock(mid_dim, num_heads=8, cond_dim=c_dim)
        
        # --- Decoder (Up Path) ---
        for dim_in, dim_out in reversed(in_out):
            self.ups.append(nn.ModuleList([
                # 【修正点】：输入维度改为 dim_out + dim_in
                nn.Linear(dim_out + dim_in, dim_out), 
                
                DiTBlock(dim_out, num_heads=4, cond_dim=c_dim),
                DiTBlock(dim_out, num_heads=4, cond_dim=c_dim),
                Upsample(dim_out, dim_in)
            ]))
            
        # 3. 输出层
        self.final_layer = FinalLayer(dim, input_channels, cond_dim=c_dim)
        
    def forward(self, x, t, classes, cond_drop_prob=None):
        """
        x: (Batch, Channels, Length) 或 (Batch, Length, Channels)
        t: (Batch,)
        classes: (Batch, Cond_Dim) 或 (Batch, Seq, Cond_Dim)
        """
        cond_drop_prob = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob
        batch_size = x.shape[0]
        
        # ================= 维度与数据预处理 =================
        
        # 1. 确保 x 是 (Batch, Length, Channels) -> Transformer 偏好格式
        # 假设输入特征数 self.x_embedder.in_features 对应 Channels
        in_channels = self.x_embedder.in_features
        
        # 记录原始长度，用于最后强制恢复
        original_length = x.shape[-1] if x.shape[1] == in_channels else x.shape[1]
        
        if x.ndim == 3 and x.shape[1] == in_channels: 
             # (B, C, L) -> (B, L, C)
             x = x.transpose(1, 2)
             
        # 2. 确保 classes 是 2D (B, Cond_Dim)
        if classes.ndim == 3:
            # 如果是序列，取最后一个时间步
            classes = classes[:, -1, :] 
        
        # ================= 嵌入与条件生成 =================
        
        # Data Projection
        x = self.x_embedder(x) # (B, L, Dim)
        
        # CFG Condition Dropout
        if cond_drop_prob > 0 and self.training:
            mask = torch.rand(batch_size, device=x.device) < cond_drop_prob
            # 丢弃时将标签置 0
            classes = torch.where(mask.unsqueeze(1), torch.zeros_like(classes), classes)
            
        t_emb = self.t_embedder(t)      # (B, Dim)
        y_emb = self.y_embedder(classes)# (B, Dim)
        
        # 融合条件 c，作为 AdaLN 的全局输入
        # c 的维度始终是 base_dim
        c = t_emb + y_emb 

        # ================= Encoder (Down) =================
        h = [] # Skip Connections
        
        for block1, block2, downsample in self.downs:
            x = block1(x, c)
            x = block2(x, c)
            h.append(x)
            x = downsample(x)

        # ================= Middle =================
        x = self.mid_block1(x, c)
        x = self.mid_block2(x, c)

        # ================= Decoder (Up) =================
        for linear_fuse, block1, block2, upsample in self.ups:
            h_pop = h.pop()
            
            # 自动对齐长度 (解决 window_size=1 或奇数长度问题)
            # 比如 x (Up后) 是 2, h_pop (Skip) 是 1
            if x.shape[1] != h_pop.shape[1]:
                # Transpose -> Interpolate -> Transpose
                x = x.transpose(1, 2)
                x = F.interpolate(x, size=h_pop.shape[1], mode='nearest')
                x = x.transpose(1, 2)
            
            # 在特征维度拼接 (B, L, Dim_Curr * 2)
            x = torch.cat((x, h_pop), dim=-1) 
            
            # 融合维度回 Dim_Curr
            x = linear_fuse(x)
            
            x = block1(x, c)
            x = block2(x, c)
            x = upsample(x)

        # ================= Output =================
        
        # 强制恢复原始长度 (处理最后的 Upsample 导致的不匹配)
        if x.shape[1] != original_length:
            x = x.transpose(1, 2)
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
            x = x.transpose(1, 2)

        # 最终 AdaLN + Projection
        x = self.final_layer(x, c)
        
        # 转回 (B, C, L) 以匹配 PyTorch Conv 习惯和 Loss 计算要求
        x = x.transpose(1, 2)
        
        return x