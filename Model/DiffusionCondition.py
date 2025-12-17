import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class RectifiedFlowTrainer(nn.Module):
    def __init__(self, model, t_eps=1e-5, loss_type='mse', label_drop_prob=0.1):
        """
        基于 Flow Matching 的训练器
        """
        super().__init__()
        self.model = model
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.label_drop_prob = label_drop_prob
        
        # Logit-Normal 采样的时间分布参数 (参考 EDM / Flow Matching 论文)
        self.P_mean = -0.8
        self.P_std = 0.8

    def forward(self, x_0, labels):
        """
        x_0: 真实数据 (Batch, Seq_Len, Dim) [也就是 t=1 的状态]
        labels: 条件向量 (Batch, Seq_Len, Dim) 或 (Batch, Dim)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # 1. 采样时间步 t (Logit-Normal 分布，更关注中间过程)
        # t 的范围 [0, 1]，其中 0 代表噪声，1 代表数据
        rnd_normal = torch.randn([batch_size], device=device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        t = t / (1 + t) # Sigmoid 变换将范围映射到 (0, 1)
        # 或者简单的 Uniform 分布: t = torch.rand([batch_size], device=device)
        
        # 扩展维度以便广播: (B, 1, 1)
        t_b = t.view(batch_size, 1, 1)

        # 2. 生成噪声 (t=0 的状态)
        x_1 = torch.randn_like(x_0) # 噪声

        # 3. 构造插值轨迹 z_t (Linear Interpolation)
        # Rectified Flow 定义: z_t = t * x_0 + (1 - t) * x_1
        # t=1 -> x_0 (数据), t=0 -> x_1 (噪声)
        z_t = t_b * x_0 + (1.0 - t_b) * x_1

        # 4. 计算目标速度场 (Target Velocity)
        # v = d(z_t)/dt = x_0 - x_1
        target_v = x_0 - x_1

        # 5. 连续条件 Dropout (CFG 训练的关键)
        if self.label_drop_prob > 0:
            # 生成随机掩码: True 表示丢弃条件
            drop_mask = (torch.rand(batch_size, device=device) < self.label_drop_prob).view(batch_size, 1, 1)
            # 丢弃时将标签设为全 0
            labels_in = torch.where(drop_mask, torch.zeros_like(labels), labels)
        else:
            labels_in = labels

        # 6. 模型预测
        pred_v = self.model(z_t, t, labels_in)

        # 7. 计算损失
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_v, target_v, reduction='none')
            loss = loss.mean()
        elif self.loss_type == 'l1':
            loss = F.l1_loss(pred_v, target_v).mean()
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        return loss


class RectifiedFlowSampler(nn.Module):
    def __init__(self, model, t_eps=1e-5):
        super().__init__()
        self.model = model
        self.t_eps = t_eps

    def get_velocity(self, z, t, labels, cfg_scale):
        """
        计算经过 Classifier-Free Guidance 调整后的速度场
        """
        batch_size = z.shape[0]
        
        # ==================== 【修改开始】 ====================
        # 严格处理 t 的维度，确保它是 (Batch_Size,) 的形状
        if isinstance(t, (int, float)):
            # 如果是 Python 数值，填充为 Tensor
            t_tensor = torch.full((batch_size,), t, device=z.device, dtype=torch.float32)
        elif isinstance(t, torch.Tensor):
            if t.ndim == 0:
                # 如果是标量 Tensor (例如 tensor(0.5))，扩展为 (B,)
                t_tensor = t.unsqueeze(0).expand(batch_size)
            elif t.ndim == 1 and t.shape[0] == 1:
                # 如果是 (1,) Tensor，扩展为 (B,)
                t_tensor = t.expand(batch_size)
            else:
                # 已经是 (B,) Tensor
                t_tensor = t
        else:
            raise ValueError(f"Unsupported type for time t: {type(t)}")
        # ==================== 【修改结束】 ====================

        # 1. 有条件预测
        v_cond = self.model(z, t_tensor, labels)

        # 2. 如果启用 CFG，进行无条件预测并组合
        if cfg_scale > 0: 
            # 构造全 0 的无条件输入
            zeros_label = torch.zeros_like(labels)
            v_uncond = self.model(z, t_tensor, zeros_label)
            
            # CFG 公式: v = v_uncond + scale * (v_cond - v_uncond)
            # 这里的公式变化：scale 对应 (1 + w)
            # 如果 cfg_scale 是 w (例如 0.2)，则公式通常是 v_uncond + (1+w)*(v_cond - v_uncond) 
            # 或者 v_cond + w*(v_cond - v_uncond)
            # 这里沿用 Rectified Flow 常见的写法：
            v_final = v_uncond + (1 + cfg_scale) * (v_cond - v_uncond)
        else:
            v_final = v_cond

        return v_final

    @torch.no_grad()
    def forward(self, noise, labels, steps=50, cfg_scale=0.0, method='euler'):
        """
        采样函数 (ODE Solver)
        x_T: 初始噪声 (Batch, Seq, Dim)
        labels: 条件
        steps: 采样步数 (20-50 通常足够)
        cfg_scale: 引导强度 (相当于 w)
        method: 'euler' (快) 或 'heun' (更准)
        """
        b = noise.shape[0]
        device = noise.device
        
        # 初始化 z 为噪声 (对应 t=0)
        z = noise.clone()

        # 生成时间步序列: 0 -> 1
        time_steps = torch.linspace(0, 1, steps + 1, device=device)

        for i in range(steps):
            t_curr = time_steps[i]
            t_next = time_steps[i+1]
            dt = t_next - t_curr

            # 扩展维度以便广播计算
            dt_b = dt.view(1, 1, 1)

            if method == 'euler':
                # Euler 方法: z_{t+1} = z_t + v(z_t) * dt
                v = self.get_velocity(z, t_curr, labels, cfg_scale)
                z = z + v * dt_b
            
            elif method == 'heun':
                # Heun 方法 (改进的 Euler): 
                # 1. 预估下一步
                v1 = self.get_velocity(z, t_curr, labels, cfg_scale)
                z_guess = z + v1 * dt_b
                
                # 2. 在预估点计算速度
                v2 = self.get_velocity(z_guess, t_next, labels, cfg_scale)
                
                # 3. 取平均速度更新
                z = z + 0.5 * (v1 + v2) * dt_b

        # 最终结果 (t=1 的状态)
        return z

    # 兼容旧代码的接口
    def sample_backward(self, img_or_shape, device, label, **kwargs):
        # 简单的包装器以适应旧的调用方式
        if isinstance(img_or_shape, torch.Tensor):
            noise = img_or_shape
        else:
            noise = torch.randn(img_or_shape).to(device)
            
        return self.forward(noise, label, steps=50, cfg_scale=self.model.w if hasattr(self.model, 'w') else 1.0)