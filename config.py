import os
import torch
import numpy as np

# 项目路径配置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "数据")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

# 确保目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 数据集配置
TRAIN_DATA_DIRS = [
    os.path.join(DATA_DIR, "navtrain_current_4"),
    os.path.join(DATA_DIR, "navtrain_current_5"),
]
VAL_DATA_DIRS = [os.path.join(DATA_DIR, "navtrain_current_4_part")]
TEST_DATA_DIRS = [os.path.join(DATA_DIR, "navtrain_current_6")]

# nuPlan数据集配置
USE_NUPLAN = True  # 是否使用nuPlan数据集
NUPLAN_DATA_ROOT = os.path.join(DATA_DIR, "nuplan")
NUPLAN_MAP_ROOT = os.path.join(NUPLAN_DATA_ROOT, "maps")
NUPLAN_DB_FILES = {
    'train': [
        os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1", "train", "2021.05.12.22.00.10_veh-35_01008_01518.db"),
        os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1", "train", "2021.06.09.17.23.18_veh-38_01151_01532.db"),
    ],
    'val': [
        os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1", "val", "2021.10.11.08.31.07_veh-38_00773_01140.db"),
    ],
    'test': [
        os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1", "test", "2021.10.06.07.26.10_veh-52_00006_00398.db"),
    ]
}

# 模型配置
PERCEPTION_CONFIG = {
    # Transfuser模型配置
    'lidar_backbone': 'resnet34',  # LiDAR主干网络
    'image_backbone': 'resnet34',  # 图像主干网络
    'transformer_layers': 4,       # Transformer层数
    'transformer_heads': 8,        # Transformer注意力头数
    'transformer_dim': 512,        # Transformer模型维度
}

# 轨迹解码器配置
TRAJECTORY_CONFIG = {
    'num_clusters': 8192,          # 词汇表大小(V8192)
    'future_horizon': 4.0,         # 未来预测时间范围(秒)
    'trajectory_frequency': 10,    # 轨迹点频率(Hz)
    'total_timesteps': 40,         # 轨迹总时间步
    'latent_dim': 256,             # 潜在空间维度
    'transformer_layers': 6,       # Transformer层数
    'transformer_heads': 8,        # Transformer注意力头数
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,              # 批大小 - 减小批大小以减少内存压力，提高吞吐量
    'num_gpus': 3,                 # 使用GPU数量，更新为3张卡
    'learning_rate': 1e-4,         # 学习率
    'weight_decay': 0.0,           # 权重衰减
    'num_epochs': 20,              # 训练轮数
    'lr_scheduler_step': 5,        # 学习率调整步长
    'lr_scheduler_gamma': 0.5,     # 学习率调整系数
    'num_workers': 8,              # 每个进程的数据加载工作线程数
    'prefetch_factor': 2,          # 数据预取因子
    'gradient_accumulation_steps': 4, # 梯度累积步数，模拟更大批量
    'mixed_precision': True,       # 使用混合精度训练
    'cudnn_benchmark': True,       # 启用cuDNN基准测试以优化性能
}

# 蒸馏配置
DISTILLATION_CONFIG = {
    'weight_im': 1.0,              # 模仿损失权重
    'weight_kd': 1.0,              # 知识蒸馏损失权重
    'w1': 0.1,                     # 模仿分数权重
    'w2': 0.5,                     # NC分数权重基础值（实际权重将动态计算）
    'w3': 0.5,                     # DAC分数权重
    'w4': 5.0,                     # TTC/C/EP组合权重
}

# PDM分数配置
PDM_CONFIG = {
    'ttc_weight': 5,               # TTC权重
    'c_weight': 2,                 # 舒适度权重
    'ep_weight': 5,                # 自我进展权重
}

# GPU设置
def setup_device():
    """设置并返回可用的计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 启用cuDNN自动优化器
        if TRAINING_CONFIG.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
            print("已启用cuDNN基准测试以优化性能")
        print(f"使用CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    return device

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 

# 检查是否有GPU可用（包括CUDA和MPS）
def has_gpu():
    """检查是否有GPU可用（包括CUDA和MPS）"""
    return torch.cuda.is_available() or (hasattr(torch, 'mps') and torch.backends.mps.is_available()) 