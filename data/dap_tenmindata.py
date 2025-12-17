import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from numpy.lib.stride_tricks import sliding_window_view
from args import args
class DAPTenMindata:
    def __init__(self, sequence_length, data_path=None):
        self.sequence_length = sequence_length
        self.train_ratio = 0.8  # 训练集比例，剩余为测试集
        self.data_path = data_path
        # 1. 路径处理
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'dap_tenmindata_7_202101_cleaned.csv')
        
        # 2. 定义列名
        test_cols = ['id', 'cycle', 'Status', 'WindSpeedMean', 'WindDirectionMean', 'ActivePowerMean', 'ReActivePowerMean', 
                    'ErrCode', 'MainBearingSpeedMean', 'GeneratorSpeedMean', 'MainBearingTempFrontMean', 'MainBearingTempBackMean', 
                    'GearboxDEBearingTempMean', 'GearboxNDEBearingTempMean', 'GearboxOilSumpTempMean', 'GeneratorDEBearingTempMean', 
                    'GeneratorNDEBearingTempMean', 'GeneratorWindingTempUMean', 'GeneratorWindingTempVMean', 'GeneratorWindingTempWMean', 
                    'AmbientTempMean', 'NacelleTempMean', 'PitchPosition1Mean', 'PitchPosition2Mean', 'PitchPosition3Mean', 'YawErrorMean', 
                    'GridPhaseCurrentABMean', 'GridPhaseCurrentBCMean', 'GridPhaseCurrentCAMean']
        
        self.columns = ['DataTime', 'id', 'Status', 'WindSpeedMean', 'WindDirectionMean', 'ActivePowerMean', 
                        'ReActivePowerMean', 'MainBearingSpeedMean', 'GeneratorSpeedMean', 'MainBearingTempFrontMean', 
                        'MainBearingTempBackMean', 'GearboxDEBearingTempMean', 'GearboxNDEBearingTempMean', 
                        'GearboxOilSumpTempMean', 'GeneratorDEBearingTempMean', 'GeneratorNDEBearingTempMean', 
                        'GeneratorWindingTempUMean', 'GeneratorWindingTempVMean', 'GeneratorWindingTempWMean', 
                        'AmbientTempMean', 'NacelleTempMean', 'PitchPosition1Mean', 'PitchPosition2Mean', 
                        'PitchPosition3Mean', 'YawErrorMean', 'GridPhaseCurrentABMean', 'GridPhaseCurrentBCMean', 
                        'GridPhaseCurrentCAMean', 'GeneratorTorqueMean', 'ErrCode'] if not data_path.endswith('test.csv') else test_cols
        
        self.setting_columns = ["WindSpeedMean","WindDirectionMean","ActivePowerMean", "ReActivePowerMean","AmbientTempMean","NacelleTempMean","PitchPosition1Mean"]
        
        self.feature_columns = ["MainBearingSpeedMean","GeneratorSpeedMean","MainBearingTempFrontMean",
            "MainBearingTempBackMean","GearboxDEBearingTempMean","GearboxNDEBearingTempMean","GearboxOilSumpTempMean",
            "GeneratorDEBearingTempMean","GeneratorNDEBearingTempMean","GeneratorWindingTempUMean","GeneratorWindingTempVMean",
            "GeneratorWindingTempWMean","YawErrorMean",
            "GridPhaseCurrentABMean","GridPhaseCurrentBCMean","GridPhaseCurrentCAMean"]  # "PitchPosition1Mean","PitchPosition2Mean","PitchPosition3Mean",
        
        # 3. 读取与预处理
        print(f"Loading data from {data_path}")
        raw_data = pd.read_csv(data_path, header=0, index_col=False, delimiter=',')

        # 采样 (50% 数据)
        grouped = raw_data.groupby('id', group_keys=False).apply(lambda x: x.head(int(len(x))))
        
        # 排序
        self.data = grouped.sort_values(by=['id', 'DataTime']).reset_index(drop=True)
        self.engine_ids = self.data['id'].unique()
        
        # 4. 统计极值
        args.max_normal = self.data[self.feature_columns].max(axis=0).to_numpy()
        args.min_normal = self.data[self.feature_columns].min(axis=0).to_numpy()
        args.max_label = self.data[self.setting_columns].max(axis=0).to_numpy()
        args.min_label = self.data[self.setting_columns].min(axis=0).to_numpy()
        
        # 5. 全局归一化
        # 由于要对同一ID内部随机划分，为了保证特征分布一致性，这里采用全量数据的MinMax参数
        self.scaler = MinMaxScaler()
        target_cols = self.feature_columns + self.setting_columns
        self.data[target_cols] = self.scaler.fit_transform(self.data[target_cols])
        
        # ==========================================
        # 6. 【核心】生成窗口并进行随机划分
        # ==========================================
        print("Processing windows and splitting randomly within each ID...")
        
        train_feats_list = []
        train_lbls_list = []
        
        # 存储测试集字典: {id: {'feature': tensor, 'label': tensor}}
        self.test_data_map = {}
        
        for eid in self.engine_ids:
            # 获取当前引擎的所有数据
            engine_mask = self.data['id'] == eid
            engine_feats_raw = self.data.loc[engine_mask, self.feature_columns].values
            engine_lbls_raw = self.data.loc[engine_mask, self.setting_columns].values
            
            # 生成滑动窗口 (N, Seq, Dim)
            win_feats = self._create_windows(engine_feats_raw) # (Batch, Seq, FeatDim)
            win_lbls = self._create_windows(engine_lbls_raw)   # (Batch, Seq, LabelDim)
            
            if win_feats is None:
                continue
                
            num_samples = win_feats.shape[0]
            
            # 生成随机索引
            indices = np.random.permutation(num_samples)
            split_idx = int(num_samples * self.train_ratio)
            
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            
            # 划分数据
            if len(train_idx) > 0:
                train_feats_list.append(win_feats[train_idx])
                train_lbls_list.append(win_lbls[train_idx])
            
            if len(test_idx) > 0:
                # 存入测试字典，方便按 ID 获取
                self.test_data_map[eid] = {
                    'feature': torch.from_numpy(win_feats[test_idx].astype(np.float32)),
                    'label': torch.from_numpy(win_lbls[test_idx].astype(np.float32))
                }
        
        # 聚合所有训练数据
        if train_feats_list:
            self.train_features = torch.from_numpy(np.concatenate(train_feats_list, axis=0).astype(np.float32))
            self.train_labels = torch.from_numpy(np.concatenate(train_lbls_list, axis=0).astype(np.float32))
        else:
            self.train_features = torch.empty(0)
            self.train_labels = torch.empty(0)
            
        print(f"Total Train Samples: {self.train_features.shape[0]}")
        print(f"Test Set available for {len(self.test_data_map)} engines.")

    def _create_windows(self, matrix_data):
        """
        将 2D 矩阵转换为 3D 滑动窗口
        Input: (Total_Time, Dim)
        Output: (Batch, Seq_Len, Dim)
        """
        if len(matrix_data) < self.sequence_length:
            return None
        # axis=0 表示在时间维度上滑动
        windows = sliding_window_view(matrix_data, window_shape=self.sequence_length, axis=0)
        # 调整维度: (Batch, Dim, Seq_Len) -> (Batch, Seq_Len, Dim)
        windows = np.moveaxis(windows, -1, 1)
        return windows

    # =========================================================================
    # 接口适配：为了兼容 main.py，通过返回特定对象或标志位来传递预处理好的 Tensor
    # =========================================================================

    def get_train_data(self):
        """
        返回预处理好的训练特征和标签的元组。
        注意：这与之前的返回 DataFrame 不同，需要配合下面的 slice 函数使用。
        为了兼容性，我们返回一个包含 'TRAIN' 标志的对象，或者直接返回 self。
        这里我们返回 self 实例本身，配合 get_feature_slice 的重载。
        """
        return "TRAIN_SET_FLAG"

    def get_test_data(self, test_id=46):
        """
        返回指定 ID 的测试数据字典或标志
        """
        if test_id not in self.test_data_map:
            print(f"Warning: No test data found for ID {test_id} (maybe split resulted in 0 samples?)")
            return None
        return {"id": test_id, "type": "TEST_SET_FLAG"}

    def get_feature_slice(self, data):
        """
        获取特征切片。
        根据 data 的类型判断是返回训练集还是测试集。
        """
        if data == "TRAIN_SET_FLAG":
            return self.train_features
        
        if isinstance(data, dict) and data.get("type") == "TEST_SET_FLAG":
            test_id = data["id"]
            return self.test_data_map[test_id]['feature']
            
        # 兼容旧逻辑（如果传入的是 DataFrame，虽然在这个新类中不应该发生）
        if isinstance(data, pd.DataFrame):
            return self._get_slice_by_id_legacy(data, self.feature_columns)
            
        raise ValueError(f"Unknown data type passed to get_feature_slice: {type(data)}")

    def get_label_slice(self, data):
        """
        获取标签切片。
        """
        if data == "TRAIN_SET_FLAG":
            return self.train_labels
        
        if isinstance(data, dict) and data.get("type") == "TEST_SET_FLAG":
            test_id = data["id"]
            return self.test_data_map[test_id]['label']
            
        if isinstance(data, pd.DataFrame):
            return self._get_slice_by_id_legacy(data, self.setting_columns)

        raise ValueError(f"Unknown data type passed to get_label_slice: {type(data)}")

    def get_test_label_slice(self, data):
        """兼容接口，直接调用 get_label_slice"""
        return self.get_label_slice(data)

    def _get_slice_by_id_legacy(self, data, columns):
        """
        保留旧的切片逻辑以防万一
        """
        slice_list = []
        current_ids = data['id'].unique()
        for eid in current_ids:
            engine_vals = data.loc[data['id'] == eid, columns].values
            windows = self._create_windows(engine_vals)
            if windows is not None:
                slice_list.append(windows)
        if not slice_list: return torch.empty(0)
        return torch.from_numpy(np.concatenate(slice_list, axis=0).astype(np.float32))

    def get_last_data_slice(self, data):
        """
        注意：在随机采样模式下，'last slice' 概念变得模糊。
        这里我们返回每个 ID 原始数据的最后一段，用于可视化或其他用途。
        """
        feature_list = []
        label_list = []
        for eid in self.engine_ids:
            engine_mask = self.data['id'] == eid
            engine_feats = self.data.loc[engine_mask, self.feature_columns].values
            engine_lbls = self.data.loc[engine_mask, self.setting_columns].values
            
            if len(engine_feats) < self.sequence_length:
                continue # 忽略太短的数据
            
            # 截取最后一段 seq_len
            feature_list.append(engine_feats[-self.sequence_length:])
            label_list.append(engine_lbls[-self.sequence_length:])
            
        return (
            torch.from_numpy(np.array(feature_list, dtype=np.float32)), 
            torch.from_numpy(np.array(label_list, dtype=np.float32))
        )