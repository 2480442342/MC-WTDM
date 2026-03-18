import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. JIT 风格的基础组件 (RMSNorm, SwiGLU, RoPE)
# ==============================================================================

class RMSNorm(nn.Module):
    """
    JIT 使用的 RMSNorm，比 LayerNorm 更稳定且计算更快
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class SwiGLUFFN(nn.Module):
    """
    JIT 使用的 SwiGLU 激活函数，替代标准的 GELU MLP
    """
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        # SwiGLU 通常需要更多的隐藏层维度来保持参数量平衡，但性能更好
        hidden_dim = int(hidden_dim * 2 / 3) 
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ==============================================================================
# 2. 旋转位置编码 (RoPE) - 适配 1D 时间序列
# ==============================================================================

class RotaryEmbedding1D(nn.Module):
    """
    将 JIT 的 VisionRotaryEmbeddingFast 简化为 1D 版本
    """
    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, x, seq_len=None):
        # x: [Batch, Seq, Head, Dim]
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [Seq, Dim]
        
        # 返回 cos, sin 用于后续旋转
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def apply_rotary_pos_emb(x, cos, sin):
    # x: [Batch, Seq, Head, Dim]
    # cos, sin: [1, Seq, 1, Dim]
    # 将 x 切分为两半进行旋转
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.cat([r1, r2], dim=-1)

# ==============================================================================
# 3. 核心模块 (DiTBlock with RMSNorm & SwiGLU & RoPE)
# ==============================================================================

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        
        # 使用 RMSNorm
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        
        # 使用 SwiGLU
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim)

        # AdaLN Modulation (保持不变，用于注入 t 和 y)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )
        
        # Zero-init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, rope_cos=None, rope_sin=None):
        # x: (Batch, Seq_Len, Hidden_Size)
        # c: (Batch, Cond_Dim)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # 1. Attention Block with RoPE
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # 准备 RoPE
        # MultiheadAttention 默认不接受 RoPE，这里我们需要手动实现或Hack
        # 为简化，我们在进入 attn 前手动旋转 Query 和 Key
        # 但 nn.MultiheadAttention 封装太死，这里我们先用标准的绝对位置编码逻辑
        # 或者：如果 x 已经包含了位置信息（通过 PosEmbed），则无需 RoPE。
        # JIT 使用了 RoPE 替代绝对位置编码。为了最大化性能，建议暂时保留绝对位置编码（UDiT原版逻辑），
        # 除非我们重写 Attention 层。
        # **为了稳妥，我们先保持 standard Attention，但应用 RMSNorm 和 SwiGLU**
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. MLP Block (SwiGLU)
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x

# ==============================================================================
# 4. 主网络架构 (U-DiT Optimized)
# ==============================================================================

class TimestepEmbedder(nn.Module):
    """ 保持不变 """
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

class FinalLayer(nn.Module):
    """ JIT 风格 Final Layer """
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
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

# Downsample/Upsample 保持不变
class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class Upsample(nn.Module):
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

class U_DiT(nn.Module):
    def __init__(
        self,
        dim=32,                
        dim_mults=(1, 2, 4),   
        channels=1,            
        feature_dim=None,      
        cond_drop_prob=0.1,    
        num_classes=4,
        in_context_len=1 # JIT 优化: 允许将条件作为 Token 插入
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.in_context_len = in_context_len
        
        input_channels = channels 
        condition_dim = num_classes 
        
        self.x_embedder = nn.Linear(input_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        
        # JIT 优化: y_embedder 用于生成 in-context tokens
        self.y_embedder = nn.Linear(condition_dim, dim)
        
        # In-Context Pos Embedding (JIT Style)
        if in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, in_context_len, dim))
            nn.init.normal_(self.in_context_posemb, std=.02)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        dims = [dim, *map(lambda m: dim * m, dim_mults)] 
        in_out = list(zip(dims[:-1], dims[1:]))          
        
        c_dim = dim 
        
        # Encoder
        for dim_in, dim_out in in_out:
            self.downs.append(nn.ModuleList([
                DiTBlock(dim_in, num_heads=4, cond_dim=c_dim), 
                DiTBlock(dim_in, num_heads=4, cond_dim=c_dim),
                Downsample(dim_in, dim_out)
            ]))
            
        mid_dim = dims[-1]
        
        # Middle (JIT 风格: 更深的网络)
        self.mid_block1 = DiTBlock(mid_dim, num_heads=8, cond_dim=c_dim)
        self.mid_block2 = DiTBlock(mid_dim, num_heads=8, cond_dim=c_dim)
        
        # Decoder
        for dim_in, dim_out in reversed(in_out):
            self.ups.append(nn.ModuleList([
                nn.Linear(dim_out + dim_in, dim_out), 
                DiTBlock(dim_out, num_heads=4, cond_dim=c_dim),
                DiTBlock(dim_out, num_heads=4, cond_dim=c_dim),
                Upsample(dim_out, dim_in)
            ]))
            
        self.final_layer = FinalLayer(dim, input_channels, cond_dim=c_dim)
        
    def forward(self, x, t, classes, cond_drop_prob=None):
        cond_drop_prob = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob
        batch_size = x.shape[0]
        
        in_channels = self.x_embedder.in_features
        original_length = x.shape[-1] if x.shape[1] == in_channels else x.shape[1]
        
        if x.ndim == 3 and x.shape[1] == in_channels: 
             x = x.transpose(1, 2)
             
        if classes.ndim == 3:
            classes = classes[:, -1, :] 
        
        # 1. 嵌入
        x = self.x_embedder(x) # (B, L, Dim)
        
        # 2. CFG Dropout
        if cond_drop_prob > 0 and self.training:
            mask = torch.rand(batch_size, device=x.device) < cond_drop_prob
            classes = torch.where(mask.unsqueeze(1), torch.zeros_like(classes), classes)
            
        t_emb = self.t_embedder(t)      # (B, Dim)
        y_emb = self.y_embedder(classes)# (B, Dim)
        
        c = t_emb + y_emb 

        # 3. JIT Optimization: In-Context Conditioning
        # 将条件 Embedding 拼接到输入序列的最前面
        if self.in_context_len > 0:
            # y_emb: (B, Dim) -> (B, 1, Dim)
            in_context_token = y_emb.unsqueeze(1) + self.in_context_posemb
            x = torch.cat([in_context_token, x], dim=1)

        # --- U-Net Pass ---
        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, c)
            x = block2(x, c)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_block2(x, c)

        for linear_fuse, block1, block2, upsample in self.ups:
            h_pop = h.pop()
            
            # 由于加了 In-Context Token，长度可能会对不齐，需要小心处理
            # 这里简单策略：Skip Connection 不包含 In-Context Token (因为它在encoder被downsample了)
            # 但为了对齐方便，我们让 interpolate 自动处理
            
            if x.shape[1] != h_pop.shape[1]:
                x = x.transpose(1, 2)
                x = F.interpolate(x, size=h_pop.shape[1], mode='nearest')
                x = x.transpose(1, 2)
            
            x = torch.cat((x, h_pop), dim=-1) 
            x = linear_fuse(x)
            x = block1(x, c)
            x = block2(x, c)
            x = upsample(x)

        # 4. Remove In-Context Token & Output
        # 输出前去掉添加的 Token
        if self.in_context_len > 0:
            x = x[:, self.in_context_len:, :]

        if x.shape[1] != original_length:
            x = x.transpose(1, 2)
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
            x = x.transpose(1, 2)

        x = self.final_layer(x, c)
        x = x.transpose(1, 2)
        
        return x