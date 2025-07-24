import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import TRAJECTORY_CONFIG

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 定义MLP层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        参数:
            tgt: 目标序列 [tgt_len, batch_size, d_model]
            memory: 编码器输出 [src_len, batch_size, d_model]
            tgt_mask: 目标序列掩码 [tgt_len, tgt_len]
            memory_mask: 记忆掩码 [tgt_len, src_len]
        """
        output = self.transformer_decoder(
            tgt, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask
        )
        return output


class TrajectoryDecoder(nn.Module):
    """轨迹解码器，实现论文中的Trajectory Decoder部分"""
    def __init__(self, config=None):
        super(TrajectoryDecoder, self).__init__()
        if config is None:
            config = TRAJECTORY_CONFIG
        
        # 配置参数
        self.num_clusters = config.get('num_clusters', 8192)  # 词汇表大小K
        self.latent_dim = config.get('latent_dim', 256)
        self.transformer_layers = config.get('transformer_layers', 6)
        self.transformer_heads = config.get('transformer_heads', 8)
        self.total_timesteps = config.get('total_timesteps', 40)
        
        # 规划词汇表（在初始化时需要加载或生成）
        self.vocabulary = torch.zeros(self.num_clusters, self.total_timesteps, 3)
        
        # 词汇表嵌入
        self.vocabulary_embedding = nn.Embedding(self.num_clusters, self.latent_dim)
        
        # 环境融合 - 修复维度不匹配问题：将输入维度从256改为512以匹配perception_network的输出
        self.env_proj = nn.Linear(512, self.latent_dim)
        
        # 查询生成
        self.query_proj = MLP(
            input_dim=self.latent_dim,
            hidden_dim=self.latent_dim * 2,
            output_dim=self.latent_dim,
            num_layers=2
        )
        
        # Transformer解码器
        self.transformer = TransformerDecoder(
            d_model=self.latent_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.latent_dim * 4,
            num_layers=self.transformer_layers
        )
        
        # 轨迹预测头
        self.trajectory_head = nn.Linear(self.latent_dim, self.num_clusters)
        
        # 多目标Hydra头
        self.hydra_heads = nn.ModuleDict({
            'im': MLP(self.latent_dim, self.latent_dim, 1),  # 模仿分数
            'NC': MLP(self.latent_dim, self.latent_dim, 1),  # 无碰撞
            'DAC': MLP(self.latent_dim, self.latent_dim, 1),  # 可行驶区域合规性
            'TTC': MLP(self.latent_dim, self.latent_dim, 1),  # 到碰撞时间
            'C': MLP(self.latent_dim, self.latent_dim, 1),    # 舒适度
            'EP': MLP(self.latent_dim, self.latent_dim, 1)    # 自我进展
        })
        
    def set_vocabulary(self, vocabulary):
        """设置规划词汇表"""
        assert vocabulary.shape[0] == self.num_clusters, f"词汇表大小不匹配: {vocabulary.shape[0]} vs {self.num_clusters}"
        assert vocabulary.shape[1] == self.total_timesteps, f"时间步不匹配: {vocabulary.shape[1]} vs {self.total_timesteps}"
        self.vocabulary = vocabulary
        
    def generate_queries(self, env_tokens, k=1):
        """生成轨迹查询"""
        batch_size = env_tokens.size(0)
        
        # 环境标记投影
        env_emb = self.env_proj(env_tokens)  # [batch_size, latent_dim]
        
        # 生成k个轨迹查询
        queries = []
        for i in range(k):
            query = self.query_proj(env_emb)  # [batch_size, latent_dim]
            queries.append(query)
            
        # 拼接查询 [k, batch_size, latent_dim]
        queries = torch.stack(queries, dim=0)
        
        return queries
    
    def compute_imitation_loss(self, logits, target_trajectory):
        """计算模仿损失（公式8）"""
        batch_size = logits.size(0)
        
        
        target = target_trajectory.unsqueeze(1)  # [batch_size, 1, timesteps, 3]
        vocab = self.vocabulary.unsqueeze(0)     # [1, num_clusters, timesteps, 3]
        
        # 计算L2距离 [batch_size, num_clusters]
        l2_dist = torch.sqrt(torch.sum(
            torch.pow(target - vocab, 2), dim=[-2, -1]
        ))
        
        # 将L2距离转换为模仿目标概率（公式9）
        temperature = 0.1  # 温度参数，控制概率分布的平滑程度
        sim_exp = torch.exp(-l2_dist / temperature)
        y_i = sim_exp / torch.sum(sim_exp, dim=1, keepdim=True)  # 归一化
        
        # 计算交叉熵损失（公式8）
        loss = -torch.sum(y_i * F.log_softmax(logits, dim=1)) / batch_size
        
        return loss
    
    def forward(self, env_tokens):
        """前向传播，生成轨迹和评估指标分数"""
        batch_size = env_tokens.size(0)
        
        # 生成轨迹查询
        queries = self.generate_queries(env_tokens)  # [1, batch_size, latent_dim]
        
        # 为Transformer解码准备环境记忆
        # 确保环境记忆和查询具有相同的特征维度
        env_emb = self.env_proj(env_tokens)  # [batch_size, latent_dim]
        memory = env_emb.unsqueeze(0)  # [1, batch_size, latent_dim]
        
        # Transformer解码
        decoded_features = self.transformer(queries, memory)  # [1, batch_size, latent_dim]
        decoded_features = decoded_features.squeeze(0)  # [batch_size, latent_dim]
        
        # 预测轨迹词汇分布
        logits = self.trajectory_head(decoded_features)  # [batch_size, num_clusters]
        
        # 使用softmax生成正确的分布概率
        probs = F.softmax(logits, dim=1)
        
        # 选择最佳轨迹(使用argmax)和次佳轨迹(使用top-k)
        k_values = 5
        best_idx = torch.argmax(logits, dim=1)  # [batch_size]
        _, topk_indices = torch.topk(logits, k=k_values, dim=1)  # [batch_size, k]
        
        # 获取最佳轨迹
        best_trajectories = self.vocabulary[best_idx]  # [batch_size, timesteps, 3]
        
        # 获取top-k轨迹
        topk_trajectories = []
        for i in range(k_values):
            topk_idx = topk_indices[:, i]
            topk_trajectory = self.vocabulary[topk_idx]  # [batch_size, timesteps, 3]
            topk_trajectories.append(topk_trajectory)
        
        # 预测各指标分数
        metric_scores = {}
        for metric_name, head in self.hydra_heads.items():
            # 确保分数计算的数值稳定性
            score = head(decoded_features).squeeze(-1)  # [batch_size]
            # 应用sigmoid激活函数，并确保结果在有效范围内
            score = torch.clamp(torch.sigmoid(score), min=0.01, max=0.99)
            metric_scores[metric_name] = score
        
        # 组合输出结果
        result = {
            'logits': logits,
            'probs': probs,
            'best_idx': best_idx,
            'best_trajectories': best_trajectories,
            'topk_indices': topk_indices,
            'topk_trajectories': topk_trajectories,
            'decoded_features': decoded_features,
            'metric_scores': metric_scores
        }
        
        return result
    
    def compute_distillation_loss(self, features, target_scores):
        """计算知识蒸馏损失（公式10）"""
        batch_size = features.size(0)
        
        # 为每个指标计算二元交叉熵损失
        losses = {}
        total_loss = 0
        
        for metric_name, head in self.hydra_heads.items():
            if metric_name in target_scores:
                # 预测分数（不应用sigmoid，让binary_cross_entropy_with_logits来处理）
                pred_logits = head(features).squeeze(-1)  # [batch_size]
                
                # 目标分数
                target = target_scores[metric_name]  # [batch_size]
                
                # 使用binary_cross_entropy_with_logits，它在内部应用sigmoid并且对autocast安全
                loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='sum') / batch_size
                losses[metric_name] = loss
                total_loss += loss
        
        return total_loss, losses
    
    def to(self, *args, **kwargs):
        """重写to方法，确保vocabulary张量随着模型一起移动到指定设备"""
        self.vocabulary = self.vocabulary.to(*args, **kwargs)
        return super().to(*args, **kwargs)


def build_trajectory_decoder(config=None):
    """构建轨迹解码器"""
    model = TrajectoryDecoder(config)
    return model 