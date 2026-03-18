import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
import wandb

# 项目内部导入
import data.dap_tenmindata as DAPTenMindata
from Model.TrainCondition import train, sample
from args import args
from utils import wandb_record

# 设置 WandB 为离线模式
os.environ["WANDB_MODE"] = "offline"

from Parameters.Compare_parameters import count_parameters

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # 1. 参数默认配置 (如果未通过命令行传入)
    # -------------------------------------------------------------------------
    if len(sys.argv) == 1:
        print('------- No Prompt, Using Default Arguments --------')
        args.epoch = 50
        args.dataset = 'WT' # 仅作为标识，实际数据由 test_file_path 决定
        args.lr = 2e-3
        args.sample_type = 'ddpm' # ddim, ddpm
        args.state = 'train' # train, sample
        
        # 数据路径配置
        args.window_size = 30  
        # ==========================================
        
        args.input_size = 1
        
        # 你的数据文件路径
        args.test_file_path = os.path.join('data', 'dap_tenmindata_7_202101_cleaned.csv') 
        args.id_num = 46
        args.output_size = 7
    # -------------------------------------------------------------------------
    # 2. 数据集初始化与加载
    # -------------------------------------------------------------------------
    # 实例化数据集类 (此时只读取一次 CSV)
    datasets = DAPTenMindata.DAPTenMindata(
        sequence_length=args.window_size,
        data_path=args.test_file_path
    )

    # 1. 获取训练数据张量
    train_data_flag = datasets.get_train_data()
    train_tensor = datasets.get_feature_slice(train_data_flag)
    train_label_tensor = datasets.get_label_slice(train_data_flag)
    
    # 获取测试数据 (针对特定 ID)
    test_df = datasets.get_test_data(test_id=args.id_num)
    test_label_tensor = datasets.get_test_label_slice(test_df)
    
    print(f'Train Tensor Shape: {train_tensor.shape}')      # 预期 (N, 30, 17)
    print(f'Train Label Shape: {train_label_tensor.shape}') # 预期 (N, 30, 7)
    
    # 2. 从真实的 Tensor Shape 中读取维度，更新 args
    args.feature_columns_length = train_tensor.shape[-1] # 这里会自动变成 17
    args.settings_length = train_label_tensor.shape[-1]  # 这里会自动变成 7
    
    args.input_size = args.feature_columns_length  # 模型输入维度 (17)
    args.output_size = args.settings_length        # 模型输出/条件维度 (7)

    print(f"Data Dimensions Confirmed. Input Features: {args.input_size}, Output Labels: {args.output_size}")

    # -------------------------------------------------------------------------
    # 3. 定义保存路径 (依赖上述维度参数)
    # -------------------------------------------------------------------------
    now = datetime.now()
    current_date = now.strftime(r"%Y-%m-%d-%H")
    
    # 定义模型权重保存路径
    weights_dir = 'weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        
    args.model_path = os.path.join(
        weights_dir, 
        f"{current_date}_{args.dataset}_seq{args.window_size}_feat{args.feature_columns_length}.pth"
        # r'2025-12-13-11_WT_seq30_feat16.pth'
    )

    # 定义生成数据保存路径
    syn_dir = os.path.join('weights', 'syn_data_test')
    if not os.path.exists(syn_dir):
        os.makedirs(syn_dir)
        
    args.syndata_path = os.path.join(
        syn_dir, 
        f"syn_{current_date}_{args.dataset}_feat{args.feature_columns_length}.npz"
    )

    # 初始化 WandB (在 args 参数全部确定后初始化，以便记录完整配置)
    wandb.init(
        project="MC_WTDM",
        tags=['EXP-optimize'],
        config=args
    )
   
    # -------------------------------------------------------------------------
    # 5. 模型训练与采样流程
    # -------------------------------------------------------------------------
    if args.state == "train":
        print(f"Starting Training... Saving to {args.model_path}")
        train(args, train_tensor, train_label_tensor)
        
        # 训练完成后进行采样测试
        print("Training finished. Starting Sampling...")
        sample(args, test_label_tensor)
        
    elif args.state == "sample":
        print(f"Starting Sampling (Loading from {args.model_path})...")
        # 确保模型文件存在
        if not os.path.exists(args.model_path):
            print(f"Error: Model path {args.model_path} does not exist for sampling.")
        else:
            sample(args, test_label_tensor)