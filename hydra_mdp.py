import torch
import torch.nn as nn
import torch.nn.functional as F
from perception_network import build_perception_network, PerceptionNetwork
from trajectory_decoder import build_trajectory_decoder, TrajectoryDecoder
from hydra_distillation import HydraDistillation, PDMScoreCalculator, WeightedPostprocessing, SubScoreEnsembling
from data_utils import generate_planning_vocabulary
import numpy as np
import os

class HydraMDP(nn.Module):
    """Hydra-MDP主模型，实现论文中的整体架构"""
    
    def __init__(self, perception_config=None, trajectory_config=None):
        """
        初始化Hydra-MDP模型
        
        参数:
            perception_config: 感知网络配置
            trajectory_config: 轨迹解码器配置
        """
        super(HydraMDP, self).__init__()
        
        # 构建感知网络
        self.perception_network = build_perception_network(perception_config)
        
        # 构建轨迹解码器
        self.trajectory_decoder = build_trajectory_decoder(trajectory_config)
        
        # 多目标蒸馏器
        self.distillation = HydraDistillation()
        
        # PDM分数计算器
        self.pdm_calculator = PDMScoreCalculator()
        
        # 加权后处理器
        self.postprocessor = WeightedPostprocessing()
        
        # 初始化规划词汇表
        self._initialize_vocabulary()
    
    def _initialize_vocabulary(self):
        """初始化规划词汇表"""
        # 检查保存的词汇表是否存在
        vocab_path = os.path.join('checkpoints', 'planning_vocabulary.npy')
        
        if os.path.exists(vocab_path):
            # 加载预生成的词汇表
            vocabulary = torch.from_numpy(np.load(vocab_path))
            print(f"加载规划词汇表，形状: {vocabulary.shape}")
        else:
            # 生成新的词汇表
            vocabulary = generate_planning_vocabulary(
                num_clusters=self.trajectory_decoder.num_clusters,
                save_path=vocab_path
            )
            print(f"生成新的规划词汇表，形状: {vocabulary.shape}")
        
        # 设置词汇表
        self.trajectory_decoder.set_vocabulary(vocabulary)
        # 确保可以正确加载模型时能够识别该属性
        self.register_buffer("trajectory_decoder_vocabulary", vocabulary.clone())
    
    def forward(self, lidar_bev, front_img, side_imgs=None, target=None, mode='train'):
        """
        前向传播
        
        参数:
            lidar_bev: LiDAR BEV表示 [batch_size, 6, H, W]
            front_img: 前视图像 [batch_size, 3, 224, 224]
            side_imgs: 侧视图像列表，每个元素形状 [batch_size, 3, 224, 224]
            target: 目标数据，包含轨迹和指标分数
            mode: 运行模式，'train', 'val', 或 'test'
            
        返回:
            outputs: 模型输出
        """
        batch_size = lidar_bev.size(0)
        
        # 感知网络前向传播
        perception_outputs = self.perception_network(lidar_bev, front_img, side_imgs)
        
        # 获取环境标记
        env_tokens = perception_outputs['env_tokens']
        
        # 轨迹解码器前向传播
        trajectory_outputs = self.trajectory_decoder(env_tokens)
        
        # 组合输出
        outputs = {
            'perception': perception_outputs,
            'trajectory': trajectory_outputs,
        }
        
        # 计算损失
        if mode == 'train' and target is not None:
            # 创建用于计算损失的输出字典，确保包含'features'键
            loss_outputs = {
                'logits': trajectory_outputs['logits'],
                'features': trajectory_outputs['decoded_features']  # 添加features键，对应decoded_features
            }
            loss, loss_dict = self.distillation.compute_loss(
                loss_outputs, self.trajectory_decoder, target
            )
            outputs['loss'] = loss
            outputs['loss_dict'] = loss_dict
        
        # 在验证或测试模式下，使用加权后处理选择最佳轨迹
        if mode == 'test' or mode == 'val':
            # 获取预测的指标分数
            metric_scores = trajectory_outputs['metric_scores']
            logits = trajectory_outputs['logits']
            
            # 修复trajectory_outputs输出变更后的字段名匹配
            if 'best_trajectories' in trajectory_outputs:
                best_trajectories = trajectory_outputs['best_trajectories']
                best_indices = trajectory_outputs['best_idx']
            else:
                # 选择最佳轨迹
                best_trajectories, best_indices = self.postprocessor.select_best_trajectory(
                    logits, metric_scores, self.trajectory_decoder.vocabulary
                )
            
            # 计算PDM分数
            pdm_score, sub_scores = self.pdm_calculator.calculate_pdm_score(metric_scores)
            
            # 添加到输出
            outputs['best_trajectories'] = best_trajectories
            outputs['best_indices'] = best_indices
            outputs['pdm_score'] = pdm_score
            outputs['sub_scores'] = sub_scores
            
            # 验证模式下，如果有target，计算损失
            if mode == 'val' and target is not None:
                # 创建用于计算验证损失的输出字典，确保包含'features'键
                val_loss_outputs = {
                    'logits': trajectory_outputs['logits'],
                    'features': trajectory_outputs['decoded_features']  # 添加features键，对应decoded_features
                }
                val_loss, _ = self.distillation.compute_loss(
                    val_loss_outputs, self.trajectory_decoder, target
                )
                outputs['loss'] = val_loss
        
        return outputs

    def load_state_dict(self, state_dict, strict=True):
        """
        重写load_state_dict方法，处理vocabulary键问题
        """
        modified_state_dict = state_dict.copy()
        
        # 检查是否存在trajectory_decoder.vocabulary键
        if 'trajectory_decoder.vocabulary' in modified_state_dict:
            # 获取vocabulary数据
            vocabulary = modified_state_dict.pop('trajectory_decoder.vocabulary')
            # 确保trajectory_decoder已初始化
            if hasattr(self, 'trajectory_decoder'):
                # 设置词汇表
                self.trajectory_decoder.vocabulary = vocabulary
        
        # 处理Missing key警告
        for key in list(self._buffers.keys()):
            if key == "trajectory_decoder_vocabulary" and key not in modified_state_dict:
                # 如果之前没有设置vocabulary，则不需要检查该键
                strict = False
        
        # 调用父类方法完成剩余的加载
        return super().load_state_dict(modified_state_dict, strict=strict)


class EnsembledHydraMDP:
    """集成的Hydra-MDP模型，使用多个不同配置的模型进行集成"""
    
    def __init__(self, models, weights=None):
        """
        初始化集成模型
        
        参数:
            models: Hydra-MDP模型字典
            weights: 各模型权重
        """
        self.models = models
        self.score_ensembler = SubScoreEnsembling(weights)
        self.pdm_calculator = PDMScoreCalculator()
        
    def forward(self, lidar_bev, front_img, side_imgs=None, mode='test'):
        """
        前向传播
        
        参数:
            lidar_bev: LiDAR BEV表示
            front_img: 前视图像
            side_imgs: 侧视图像列表
            mode: 运行模式
            
        返回:
            ensemble_outputs: 集成模型输出
        """
        batch_size = lidar_bev.size(0)
        
        # 存储各模型输出
        model_outputs = {}
        model_scores = {}
        
        # 运行每个模型
        for model_name, model in self.models.items():
            # 设置为评估模式
            model.eval()
            
            # 运行模型
            with torch.no_grad():
                outputs = model(lidar_bev, front_img, side_imgs, mode=mode)
            
            # 保存输出
            model_outputs[model_name] = outputs
            model_scores[model_name] = outputs['trajectory']['metric_scores']
        
        # 集成子分数
        ensemble_scores = self.score_ensembler.ensemble_scores(model_scores)
        
        # 计算集成PDM分数
        ensemble_pdm, ensemble_sub_scores = self.pdm_calculator.calculate_pdm_score(ensemble_scores)
        
        # 组合集成输出
        ensemble_outputs = {
            'model_outputs': model_outputs,
            'ensemble_scores': ensemble_scores,
            'ensemble_pdm': ensemble_pdm,
            'ensemble_sub_scores': ensemble_sub_scores
        }
        
        return ensemble_outputs


def build_hydra_mdp(perception_config=None, trajectory_config=None):
    """构建Hydra-MDP模型"""
    model = HydraMDP(perception_config, trajectory_config)
    return model


def build_hydra_mdp_variants():
    """构建不同变种的Hydra-MDP模型"""
    # Hydra-MDP-A (256 x 1024, ViT-L)
    model_a = HydraMDP(
        perception_config={
            'lidar_backbone': 'resnet34',
            'image_backbone': 'vit_l_16',
            'transformer_layers': 4,
            'transformer_heads': 8,
            'transformer_dim': 512,
        },
        trajectory_config={
            'num_clusters': 8192,
            'latent_dim': 256,
            'transformer_layers': 6,
            'transformer_heads': 8,
        }
    )
    
    # Hydra-MDP-B (512 x 2048, V2-99)
    model_b = HydraMDP(
        perception_config={
            'lidar_backbone': 'resnet34',
            'image_backbone': 'resnet50',  # 模拟V2-99
            'transformer_layers': 4,
            'transformer_heads': 8,
            'transformer_dim': 512,
        },
        trajectory_config={
            'num_clusters': 8192,
            'latent_dim': 256,
            'transformer_layers': 6,
            'transformer_heads': 8,
        }
    )
    
    # Hydra-MDP-C (混合分辨率, ViT-L + V2-99)
    model_c = HydraMDP(
        perception_config={
            'lidar_backbone': 'resnet34',
            'image_backbone': 'vit_l_16',  # ViT-L
            'transformer_layers': 6,  # 更深的Transformer
            'transformer_heads': 12,
            'transformer_dim': 768,
        },
        trajectory_config={
            'num_clusters': 8192,
            'latent_dim': 384,  # 更大的潜在维度
            'transformer_layers': 8,
            'transformer_heads': 12,
        }
    )
    
    # 创建模型字典
    models = {
        'model_a': model_a,
        'model_b': model_b,
        'model_c': model_c
    }
    
    # 创建集成模型
    weights = {'model_a': 0.3, 'model_b': 0.3, 'model_c': 0.4}
    ensemble_model = EnsembledHydraMDP(models, weights)
    
    return ensemble_model 