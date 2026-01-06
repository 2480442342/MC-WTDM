import time  # 导入时间模块
import argparse  # 导入命令行参数解析模块
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy数值计算库
import pandas as pd  # 导入Pandas数据处理库


columns = ['DataTime', 'id', 'Status', 'WindSpeedMean', 'WindDirectionMean', 'ActivePowerMean', 
                        'ReActivePowerMean', 'MainBearingSpeedMean', 'GeneratorSpeedMean', 'MainBearingTempFrontMean', 
                        'MainBearingTempBackMean', 'GearboxDEBearingTempMean', 'GearboxNDEBearingTempMean', 
                        'GearboxOilSumpTempMean', 'GeneratorDEBearingTempMean', 'GeneratorNDEBearingTempMean', 
                        'GeneratorWindingTempUMean', 'GeneratorWindingTempVMean', 'GeneratorWindingTempWMean', 
                        'AmbientTempMean', 'NacelleTempMean', 'PitchPosition1Mean', 'PitchPosition2Mean', 
                        'PitchPosition3Mean', 'YawErrorMean', 'GridPhaseCurrentABMean', 'GridPhaseCurrentBCMean', 
                        'GridPhaseCurrentCAMean', 'GeneratorTorqueMean', 'ErrCode']

# feature_columns = ['s2', 's3','s4',  's7', 's8',
#          's9',  's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
feature_columns = ["MainBearingSpeedMean","GeneratorSpeedMean","MainBearingTempFrontMean",
            "MainBearingTempBackMean","GearboxDEBearingTempMean","GearboxNDEBearingTempMean","GearboxOilSumpTempMean",
            "GeneratorDEBearingTempMean","GeneratorNDEBearingTempMean","GeneratorWindingTempUMean","GeneratorWindingTempVMean",
            "GeneratorWindingTempWMean","AmbientTempMean","NacelleTempMean","PitchPosition1Mean","PitchPosition2Mean",
            "PitchPosition3Mean","YawErrorMean","GridPhaseCurrentABMean","GridPhaseCurrentBCMean","GridPhaseCurrentCAMean"]  # 输入的特征维度必须是2的整数次幂

settings_columns = ["WindSpeedMean","WindDirectionMean","ActivePowerMean", "ReActivePowerMean"]

arg_parser = argparse.ArgumentParser(description='RANet Image classification')  # 创建参数解析器


# model arch related  # 模型架构相关参数
arch_group = arg_parser.add_argument_group('arch', '模型架构设置')  # 创建模型架构参数组
arch_group.add_argument('--model_name', type=str, default='MC-WTDM', help='模型名称，默认MC-WTDM')  # 修改默认值说明

# msdnet config  # MSDNet配置参数
arch_group.add_argument('--embedding', type=int, default=48, help='嵌入维度，默认48')  # 添加help说明
arch_group.add_argument('--hidden', type=int, default=64, help='隐藏层维度，默认64')  # 添加help说明

arch_group.add_argument('--num_head', type=int, default=1, help='多头注意力头数，默认1')  # 添加help说明
arch_group.add_argument('--num_encoder', type=int, default=1, help='编码器层数，默认1')  # 添加help说明

#TFS config  # TFS配置参数
# arch_group.add_argument('--lstm_hidden', type=int, default=64, help='LSTM隐藏层维度，默认64')  # 添加help说明
# arch_group.add_argument('--num_layers', type=int, default=1, help='LSTM层数，默认1')  # 添加help说明

# columns related  # 列相关参数
col_group = arg_parser.add_argument_group('columns', '列设置')  # 修改描述说明
col_group.add_argument('--columns', default=columns, type=list, metavar='N', help='列名称，默认由变量columns定义')  # 修改默认值说明
col_group.add_argument('--feature_columns', default=feature_columns, type=list, metavar='N', help='输入特征列名称，默认由变量feature_columns定义')  # 修改默认值说明
col_group.add_argument('--feature_columns_length', default=len(feature_columns), type=int, metavar='N', help='输入特征列长度，默认由feature_columns长度决定')  # 修改默认值说明
col_group.add_argument('--settings_columns', default=settings_columns, type=list, metavar='N', help='设置列名称，默认由变量settings_columns定义')  # 修改默认值说明
col_group.add_argument('--settings_columns_length', default=len(settings_columns), type=int, metavar='N', help='设置列长度，默认由settings_columns长度决定')  # 修改默认值说明

# training related  # 训练相关参数
optim_group = arg_parser.add_argument_group('optimization', '优化设置')  # 修改描述说明
optim_group.add_argument('--epoch', default=50, type=int, metavar='N', help='训练轮数，默认50')  # 添加help说明
optim_group.add_argument('--eva_epoch', default=30, type=int, metavar='N', help='评估轮数，默认30')  # 添加help说明

optim_group.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch大小，默认64')  # 修改默认值说明
# optim_group.add_argument('--optimizer', default='adam', choices=['sgd', 'rmsprop', 'adam'], metavar='N', help='优化器类型，默认adam，可选值包括sgd, rmsprop, adam')  # 修改help说明
optim_group.add_argument('--lr', '--learning_rate', default=2e-4, type=float, metavar='LR', help='学习率，默认0.0002')  # 修改默认值说明
# optim_group.add_argument('--lr_type', default='multistep', choices=['multistep', 'cosine', 'warmup'], metavar='LR_TYPE', help='学习率调整策略，默认multistep，可选值包括multistep, cosine, warmup')  # 修改help说明
optim_group.add_argument('--grad_clip', default=0.5, type=float, help='梯度裁剪值，默认1.0')  # 添加help说明
optim_group.add_argument('--multiplier', default=2.5, type=float, help='乘数因子，默认2.5')  # 添加help说明
optim_group.add_argument('--loss_type', default='mse', choices=['mse', 'l1'], type=str, help='损失函数类型，默认mse，可选值包括mse, l1')  # 修改help说明
optim_group.add_argument('--sample_type', default='euler', choices=['euler', 'heun'], type=str, help='采样类型，默认euler，可选值包括euler, heun')  # 修改help说明

arg_parser.add_argument('--input_size', default=1, type=int, help='输入维度，默认1')  # 添加help说明
arg_parser.add_argument('--output_size', default=3, type=int, help='输出维度，默认3')  # 修改默认值说明
arg_parser.add_argument('--window_size', default=48, type=int, help='窗口大小，默认48')  # 修改默认值说明
arg_parser.add_argument('--dropout', default=0.2, type=float, help='dropout概率，默认0.2')  # 修改默认值说明

# 数据集参数调整
arg_parser.add_argument('--dataset', default='WT', type=str, help='数据集名称，默认WT')  # 添加help说明

arg_parser.add_argument('--state', default='train', choices=['train', 'sample'], type=str, help='运行状态，默认train')  # 添加help说明

arg_parser.add_argument('--T', default=500, type=int, help='时间步长，默认500')  # 添加help说明
arg_parser.add_argument('--w', default=0.1, type=float, help='权重参数，默认0.2')  # 添加help说明

arg_parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str, help='运行设备，默认自动选择GPU或CPU')  # 修改默认值说明
arg_parser.add_argument('--model_path', default='./weights/temp.pth', type=str, help='模型保存路径，默认./weights/temp.pth')  # 添加help说明
arg_parser.add_argument('--syndata_path', default='./weights/syn_data/temp.npy', type=str, help='合成数据保存路径，默认./weights/syn_data/temp.npy')  # 添加help说明

args = arg_parser.parse_args()  # 解析命令行参数
