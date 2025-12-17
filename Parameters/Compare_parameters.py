def count_parameters(model):
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 计算可训练参数量 (requires_grad=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量 (Total params): {total_params:,}")
    print(f"可训练参数量 (Trainable params): {trainable_params:,}")
    
    # 估算模型大小 (MB)，假设是 float32 (4 bytes)
    print(f"模型大小估算 (Model Size): {total_params * 4 / 1024 / 1024:.2f} MB")