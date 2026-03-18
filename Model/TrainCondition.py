import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os
import copy

# 导入核心组件
from Model.DiffusionCondition import RectifiedFlowTrainer, RectifiedFlowSampler
from Model.UDiT import U_DiT
from Scheduler import GradualWarmupScheduler
from Parameters.Compare_parameters import count_parameters

# [保持不变] EMA 类定义
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1 - self.decay) * model_v)

def train(args, train_data, train_label):
    """
    Rectified Flow 训练函数 (集成 EMA)
    """
    device = args.device

    # 维度检查与修正
    if train_data.ndim == 3 and train_data.shape[-1] == args.input_size:
        print(f"Permuting train_data from {train_data.shape} to (Batch, Features, Seq_Len)")
        train_data = train_data.permute(0, 2, 1)
    
    # 1. 数据集构建
    train_dataset = TensorDataset(train_data.to(device), train_label.to(device))
    batch_size = getattr(args, 'batch_size', 512)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. 模型初始化
    net_model = U_DiT(
        dim=32, 
        dim_mults=(1, 2, 2), 
        cond_drop_prob=args.dropout, 
        channels=args.input_size, 
        feature_dim=args.feature_columns_length,
        num_classes=args.output_size,
    ).to(device)

    count_parameters(net_model)

    # [新增 1] 初始化 EMA 模型
    # 注意：EMA 必须在模型参数初始化之后、训练开始之前建立
    print(f"Initializing EMA with decay: {0.9999}")
    ema = EMA(net_model, decay=0.9999)

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
                optimizer.zero_grad()
                
                loss = trainer(x_batch, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
                optimizer.step()
                
                # [新增 2] 更新 EMA 权重
                # 在每次参数更新后，平滑更新影子模型
                ema.update(net_model)
                
                epoch_losses.append(loss.item())
                
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
                })

        warmup_scheduler.step()
        avg_epoch_loss = np.mean(epoch_losses)
        loss_history.append(avg_epoch_loss)
        
        wandb.log({"Diffusion_Loss": avg_epoch_loss, "Epoch": e})
        
        # 保存最佳模型
        if e > 5 and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            
            # 保存普通模型
            torch.save(net_model.state_dict(), args.model_path)
            
            # [新增 3] 保存 EMA 模型
            # 通常将 EMA 模型保存为单独的文件，推荐用于后续生成
            ema_path = args.model_path.replace('.pth', '_ema.pth')
            torch.save(ema.model.state_dict(), ema_path)
            
            print(f'--> Model Improved! Loss: {best_loss:.4f} | Saved: {args.model_path} & {ema_path}')

    # 6. 训练结束后的可视化
    plot_loss_curve(loss_history, args.model_path)
    print('Training complete!')


def sample(args, train_label=None):
    """
    Rectified Flow 采样函数 (优先加载 EMA)
    """
    if train_label is None:
        raise ValueError("Error: train_label must be provided for conditional sampling.")

    if train_label.dim() == 1:
        if train_label.shape[0] == args.output_size:
            train_label = train_label.unsqueeze(0) 
    
    device = args.device
    
    # 初始化模型结构
    net_model = U_DiT(
        dim=32,  # 必须与 train 保持一致
        dim_mults=(1, 2, 2), 
        channels=args.input_size, 
        num_classes=args.output_size,
        feature_dim=args.feature_columns_length # 补上这个参数
    ).to(args.device)


    # [修改] 智能权重加载逻辑
    # 优先寻找 _ema.pth 文件，如果找不到则回退到普通 .pth
    base_path = args.model_path
    ema_path = args.model_path.replace('.pth', '_ema.pth')
    
    load_path = base_path
    using_ema = False
    
    if os.path.exists(ema_path):
        load_path = ema_path
        using_ema = True
        print(f"[Info] Found EMA weights. Loading from: {load_path}")
    elif os.path.exists(base_path):
        print(f"[Info] EMA weights not found. Loading standard weights from: {load_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {base_path} or {ema_path}")
        
    ckpt = torch.load(load_path, map_location=args.device)
    net_model.load_state_dict(ckpt)
    net_model.eval()

    # 2. 初始化采样器
    sampler = RectifiedFlowSampler(net_model).to(device)

    # 3. 生成初始噪声
    noisy_data = torch.randn(
        size=(train_label.shape[0], args.input_size, args.window_size), 
        device=device
    )
    
    # 4. 执行采样
    cfg_scale = getattr(args, 'w', 1.5)
    sample_steps = 50
    
    if args.sample_type == 'euler':
        solve_method = 'euler'
    else:
        solve_method = 'heun'

    print(f"Sampling | Method: {solve_method}, Steps: {sample_steps}, CFG: {cfg_scale}, Using EMA: {using_ema}")

    with torch.no_grad():
        cond_tensor = train_label.to(device)
        
        sample_data = sampler(
            noise=noisy_data, 
            labels=cond_tensor, 
            steps=sample_steps, 
            cfg_scale=cfg_scale, 
            method=solve_method
        )
        
        sample_data = sample_data.cpu().numpy() 
        cond_label = train_label.cpu().numpy()

    # 5. 反归一化
    # ----------------------
    # A. 处理生成数据 (Features)
    # ----------------------
    # 形状: (Batch, 16, 30) -> 需要 reshape 统计量为 (1, 16, 1)
    if hasattr(args, 'std_normal') and hasattr(args, 'mean_normal'):
        std_feat = args.std_normal.reshape(1, -1, 1)
        mean_feat = args.mean_normal.reshape(1, -1, 1)
        denorm_data = sample_data * std_feat + mean_feat
    else:
        # 兼容旧的 MinMaxScaler (如果还没改)
        max_val = args.max_normal.reshape(1, -1, 1)
        min_val = args.min_normal.reshape(1, -1, 1)
        denorm_data = sample_data * (max_val - min_val) + min_val

    # ----------------------
    # B. 处理条件标签 (Labels)
    # ----------------------
    # 形状: 剥离出第 8 维的 timestep
    if hasattr(args, 'std_label') and hasattr(args, 'mean_label'):
        # 排除最后 1 列 timestep 的均值和标准差 (取前 7 个)
        std_lbl = args.std_label[:-1].reshape(1, 1, -1)
        mean_lbl = args.mean_label[:-1].reshape(1, 1, -1)
        
        # 分离出真实的物理控制条件 (前 7 列) 和时间标签 (最后 1 列)
        physical_label = cond_label[:, :, :-1]
        time_label = cond_label[:, :, -1:]
        
        # 只对物理控制条件进行反归一化
        denorm_physical_label = physical_label * std_lbl + mean_lbl
        
        # 重新拼接回去
        denorm_label = np.concatenate([denorm_physical_label, time_label], axis=-1)
    else:
        # 兼容旧的 MinMaxScaler
        max_lbl = args.max_label.reshape(1, 1, -1) # 或者是 (1, 1, -1)
        min_lbl = args.min_label.reshape(1, 1, -1)
        denorm_label = cond_label * (max_lbl - min_lbl) + min_lbl

    # 6. 保存数据
    print(f"Saving generated data to {args.syndata_path}")
    # 保存时建议转置一下 sample_data，让它也变成 (B, L, C) 方便后续评估，或者保持现状
    # 这里我们保持现状，但您可以根据评估代码的要求调整
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
    plt.title('Rectified Flow Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(loss_img_path)
    plt.close()