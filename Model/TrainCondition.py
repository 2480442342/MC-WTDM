import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os

# 导入核心组件
from Model.DiffusionCondition import RectifiedFlowTrainer, RectifiedFlowSampler
from Model.UDiT import U_DiT
from Scheduler import GradualWarmupScheduler

from Parameters.Compare_parameters import count_parameters

def train(args, train_data, train_label):
    """
    Rectified Flow 训练函数
    Args:
        args: 全局参数配置
        train_data: 训练特征张量 (Batch, Channels, Length)
        train_label: 训练标签张量 (Batch, Label_Dim)
    """
    device = args.device

    # 维度检查与修正 (确保是 Channel-First: Batch, 21, Seq_Len)
    if train_data.ndim == 3 and train_data.shape[-1] == args.input_size:
        print(f"Permuting train_data from {train_data.shape} to (Batch, Features, Seq_Len)")
        train_data = train_data.permute(0, 2, 1)
    
    # 1. 数据集构建
    train_dataset = TensorDataset(train_data.to(device), train_label.to(device))
    
    # 使用 args.batch_size
    batch_size = getattr(args, 'batch_size', 512)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. 模型初始化
    net_model = U_DiT(
        dim=32, 
        dim_mults=(1, 2, 2), 
        cond_drop_prob=args.dropout, # 这是模型内部特征层的 dropout，与 CFG 的 label dropout 不同
        channels=args.input_size, 
        feature_dim=args.feature_columns_length,
        num_classes=args.output_size,
    ).to(device)

    count_parameters(net_model)

    # 3. 优化器与调度器
    optimizer = optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1
    )
    
    warmup_scheduler = GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=args.multiplier,
        warm_epoch=args.epoch // 10 + 1,
        after_scheduler=cosine_scheduler
    )

    # 4. Rectified Flow 训练器初始化
    # label_drop_prob=0.1 用于实现 Classifier-Free Guidance (CFG) 的训练
    trainer = RectifiedFlowTrainer(net_model, label_drop_prob=0.1).to(device)

    # 5. 训练循环
    best_loss = float('inf')
    loss_history = []
    
    print(f"Start Training on {device} | Batch Size: {batch_size}")

    for e in range(args.epoch):
        net_model.train()
        epoch_losses = []
        
        with tqdm(dataloader, desc=f"Epoch {e+1}/{args.epoch}", dynamic_ncols=True) as pbar:
            for x_batch, y_batch in pbar:
                # x_batch: (B, C, L), y_batch: (B, Label_Dim)
                optimizer.zero_grad()
                
                # [修改点 1] 移除手动 Dropout
                # RectifiedFlowTrainer 内部已经实现了 label_drop_prob 逻辑
                # 直接传入原始标签即可
                loss = trainer(x_batch, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # 更新进度条
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
                })

        warmup_scheduler.step()
        avg_epoch_loss = np.mean(epoch_losses)
        loss_history.append(avg_epoch_loss)
        
        # 记录日志
        wandb.log({"Diffusion_Loss": avg_epoch_loss, "Epoch": e})
        
        # 保存最佳模型 (Epoch > 5 后)
        if e > 5 and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(net_model.state_dict(), args.model_path)
            print(f'--> Model Improved! Loss: {best_loss:.4f} saved to {args.model_path}')

    # 6. 训练结束后的可视化
    plot_loss_curve(loss_history, args.model_path)
    print('Training complete!')


def sample(args, train_label=None):
    """
    Rectified Flow 采样函数
    """
    if train_label is None:
        raise ValueError("Error: train_label must be provided for conditional sampling.")

    # 维度检查 (确保是 Batch 模式)
    if train_label.dim() == 1:
        if train_label.shape[0] == args.output_size:
            train_label = train_label.unsqueeze(0) 
    
    device = args.device
    
    # 1. 加载模型
    net_model = U_DiT(
        dim=32, 
        dim_mults=(1, 2, 2), 
        cond_drop_prob=args.dropout, 
        channels=args.input_size,
        feature_dim=args.feature_columns_length,
        num_classes=args.output_size,
    ).to(device)

    # 加载权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found at {args.model_path}")
        
    ckpt = torch.load(args.model_path, map_location=device)
    net_model.load_state_dict(ckpt)
    print(f"Model weights loaded from {args.model_path}")
    net_model.eval()

    # 2. 初始化采样器
    sampler = RectifiedFlowSampler(net_model).to(device)

    # 3. 生成初始噪声 (Batch, Channels, Length)
    noisy_data = torch.randn(
        size=(train_label.shape[0], args.input_size, args.window_size), 
        device=device
    )
    print(f'Sampling logic - Noisy Data Shape: {noisy_data.shape}')

    # 4. 执行采样
    # [修改点 2] 适配新的采样参数
    # cfg_scale: 对应以前的 w，控制条件强度。建议 1.0 - 2.0
    # steps: 采样步数，Rectified Flow 通常 20-50 步足够
    # method: 'euler' (快速) 或 'heun' (精确)
    
    cfg_scale = getattr(args, 'w', 1.5) # 如果 args.w 存在则使用，否则默认 1.5
    sample_steps = 50 # 或者 args.sample_steps
    
    # 简单的映射：ddpm -> euler, ddim -> heun，或者直接默认 heun
    if args.sample_type == 'euler':
        solve_method = 'euler'
    elif args.sample_type == 'heun':
        solve_method = 'heun'
    else:
        solve_method = 'heun' # 默认使用 Heun 方法，效果更好

    print(f"Sampling with Method: {solve_method}, Steps: {sample_steps}, CFG Scale: {cfg_scale}")

    with torch.no_grad():
        cond_tensor = train_label.to(device)
        
        # 调用 Sampler 的 forward
        sample_data = sampler(
            noise=noisy_data, 
            labels=cond_tensor, 
            steps=sample_steps, 
            cfg_scale=cfg_scale, 
            method=solve_method
        )
        
        # 转回 CPU Numpy
        sample_data = sample_data.cpu().numpy() 
        cond_label = train_label.cpu().numpy()

    # 5. 反归一化 (保持不变)
    max_val = args.max_normal.reshape(1, -1, 1)
    min_val = args.min_normal.reshape(1, -1, 1)
    max_lbl = args.max_label
    min_lbl = args.min_label

    denorm_data = sample_data * (max_val - min_val) + min_val
    denorm_label = cond_label * (max_lbl - min_lbl) + min_lbl

    # 6. 保存数据
    print(f"Saving generated data to {args.syndata_path}")
    np.savez(args.syndata_path, data=denorm_data, label=denorm_label)
    
    return denorm_data

def plot_loss_curve(loss_list, model_path):
    """绘制并保存 Loss 曲线 (保持不变)"""
    loss_txt_path = model_path.replace('.pth', '_loss.txt')
    loss_img_path = model_path.replace('.pth', '_loss.png')
    
    with open(loss_txt_path, 'w') as f:
        f.write(str(loss_list))
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rectified Flow Training Loss Curve') # 更新标题
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(loss_img_path)
    plt.close()