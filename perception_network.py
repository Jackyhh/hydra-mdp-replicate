import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import PERCEPTION_CONFIG

class ResNetBackbone(nn.Module):
    """ResNet主干网络，用于特征提取"""
    def __init__(self, model_name='resnet34', pretrained=True):
        super(ResNetBackbone, self).__init__()
        if model_name == 'resnet34':
            if pretrained:
                base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet34(weights=None)
        elif model_name == 'resnet50':
            if pretrained:
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                base_model = models.resnet50(weights=None)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 去掉原始ResNet的全连接层和平均池化层
        self.base_model = nn.Sequential(*list(base_model.children())[:-2])
        
        # 获取输出通道数
        if model_name == 'resnet34':
            self.out_channels = 512
        elif model_name == 'resnet50':
            self.out_channels = 2048
    
    def forward(self, x):
        x = self.base_model(x)
        return x

class ViTBackbone(nn.Module):
    """Vision Transformer主干网络，用于特征提取"""
    def __init__(self, model_name='vit_b_16', pretrained=True):
        super(ViTBackbone, self).__init__()
        weights = None
        if pretrained:
            if model_name == 'vit_b_16':
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            elif model_name == 'vit_l_16':
                weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        
        if model_name == 'vit_b_16':
            # 使用函数式API创建模型
            self.model = models.vit_b_16(weights=weights)
            self.out_channels = 768
        elif model_name == 'vit_l_16':
            self.model = models.vit_l_16(weights=weights)
            self.out_channels = 1024
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-1],  
            nn.Flatten()
        )
        
    def forward(self, x):
        # 使用特征提取器
        x = self.feature_extractor(x)
        
        # 调整维度以匹配卷积特征形状 [B, C] -> [B, C, 1, 1]
        x = x.view(x.size(0), -1, 1, 1)
        return x

class LidarBEVBackbone(nn.Module):
    """LiDAR BEV特征提取主干网络"""
    def __init__(self, in_channels=6, backbone='resnet34', pretrained=True):
        super(LidarBEVBackbone, self).__init__()
        self.in_channels = in_channels
        
        if backbone == 'resnet34':
            # 使用ResNet34作为主干网络
            if pretrained:
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet34(weights=None)
            
            # 调整第一个卷积层以适应输入通道数
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                                stride=2, padding=3, bias=False)
            
            # 获取主干网络的各个阶段
            self.layer0 = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool
            )
            self.layer1 = self.backbone.layer1
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4
            
            self.out_channels = 512
        else:
            raise ValueError(f"不支持的LiDAR主干网络: {backbone}")
    
    def forward(self, x):
        # x: [B, C, H, W]
        features = []
        
        x0 = self.layer0(x)  # 1/4
        features.append(x0)
        
        x1 = self.layer1(x0)  # 1/4
        features.append(x1)
        
        x2 = self.layer2(x1)  # 1/8
        features.append(x2)
        
        x3 = self.layer3(x2)  # 1/16
        features.append(x3)
        
        x4 = self.layer4(x3)  # 1/32
        features.append(x4)
        
        return features, x4


class TransformerLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        q = k = v = src2
        src2 = self.self_attn(q, k, v, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class ModalityFusion(nn.Module):
    """多模态融合模块，使用Transformer融合图像和LiDAR特征"""
    def __init__(self, lidar_dim, image_dim, fusion_dim, nhead, num_layers):
        super(ModalityFusion, self).__init__()
        self.lidar_dim = lidar_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim
        
        # 特征映射层
        self.lidar_projection = nn.Linear(lidar_dim, fusion_dim)
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=fusion_dim, 
            nhead=nhead,
            dim_feedforward=fusion_dim*4,
            dropout=0.1,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # 环境标记生成
        self.env_token_proj = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(self, lidar_features, image_features):
        batch_size = lidar_features.size(0)
        
        # 压平空间维度并调整维度顺序以适应batch_first=True
        lidar_flat = lidar_features.view(batch_size, self.lidar_dim, -1).permute(0, 2, 1)  # [B, L, C]
        image_flat = image_features.view(batch_size, self.image_dim, -1).permute(0, 2, 1)  # [B, L, C]
        
        # 特征映射
        lidar_embed = self.lidar_projection(lidar_flat)
        image_embed = self.image_projection(image_flat)
        
        # 拼接特征
        concat_embed = torch.cat([lidar_embed, image_embed], dim=1)  # 在序列维度拼接
        
        # Transformer编码
        fusion_features = self.transformer_encoder(concat_embed)
        
        # 分离LiDAR和图像特征
        lidar_seq_len = lidar_flat.size(1)
        lidar_fused = fusion_features[:, :lidar_seq_len, :]  # [B, L_lidar, C]
        image_fused = fusion_features[:, lidar_seq_len:, :]  # [B, L_image, C]
        
        # 生成环境标记
        env_tokens = torch.mean(fusion_features, dim=1)  # [B, C]
        env_tokens = self.env_token_proj(env_tokens)
        
        return lidar_fused, image_fused, env_tokens


class PerceptionHeads(nn.Module):
    """感知任务头，用于3D目标检测和BEV分割"""
    def __init__(self, in_channels, num_classes):
        super(PerceptionHeads, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # BEV分割头
        self.bev_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        # 3D对象检测头
        # 后续实车最好扩展成更复杂的检测头，如：车辆位姿，车道线，道路边缘，前车相对距离、速度等
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 输出 [x, y, z, w, l, h, theta, confidence, class_scores]
            nn.Conv2d(128, 9 + num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        bev_segmentation = self.bev_head(x)
        obj_detection = self.obj_head(x)
        
        return {
            'bev_segmentation': bev_segmentation,
            'obj_detection': obj_detection
        }


class PerceptionNetwork(nn.Module):
    """感知网络，实现Transfuser架构"""
    def __init__(self, config=None):
        super(PerceptionNetwork, self).__init__()
        if config is None:
            config = PERCEPTION_CONFIG
            
        # 配置参数
        self.lidar_backbone_type = config.get('lidar_backbone', 'resnet34')
        self.image_backbone_type = config.get('image_backbone', 'resnet34')
        self.transformer_layers = config.get('transformer_layers', 4)
        self.transformer_heads = config.get('transformer_heads', 8)
        self.transformer_dim = config.get('transformer_dim', 512)
        
        # LiDAR主干网络
        self.lidar_backbone = LidarBEVBackbone(
            in_channels=6,  # BEV表示的6个通道
            backbone=self.lidar_backbone_type
        )
        self.lidar_channels = self.lidar_backbone.out_channels
        
        # 图像主干网络
        if 'vit' in self.image_backbone_type:
            self.image_backbone = ViTBackbone(self.image_backbone_type)
        else:
            self.image_backbone = ResNetBackbone(self.image_backbone_type)
        self.image_channels = self.image_backbone.out_channels
        
        # 多模态融合
        self.fusion = ModalityFusion(
            lidar_dim=self.lidar_channels,
            image_dim=self.image_channels,
            fusion_dim=self.transformer_dim,
            nhead=self.transformer_heads,
            num_layers=self.transformer_layers
        )
        
        # 特征解码器 (融合特征 -> 2D空间特征)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 感知任务头
        self.perception_heads = PerceptionHeads(128, num_classes=23)  # 23类分割和检测
        
    def forward(self, lidar_bev, front_img, side_imgs=None):
        batch_size = lidar_bev.size(0)
        
        # LiDAR特征提取
        lidar_features_list, lidar_features = self.lidar_backbone(lidar_bev)
        
        # 图像特征提取
        front_img_features = self.image_backbone(front_img)
        
        # 如果有侧视图像，连接它们
        if side_imgs is not None:
            all_imgs = []
            all_imgs.append(front_img_features)
            
            for side_img in side_imgs:
                if side_img is not None:
                    side_features = self.image_backbone(side_img)
                    all_imgs.append(side_features)
            
            # 平均池化所有图像特征
            image_features = torch.mean(torch.stack(all_imgs, dim=0), dim=0)
        else:
            image_features = front_img_features
        
        # 模态融合
        lidar_fused_seq, image_fused_seq, env_tokens = self.fusion(
            lidar_features, image_features
        )
        
        # 重构2D特征图
        # 获取原始BEV特征图的空间维度
        original_h = int(lidar_features.size(2))
        original_w = int(lidar_features.size(3))
        
        # 正确地重建特征图
        lidar_fused = lidar_fused_seq.permute(0, 2, 1).contiguous().view(
            batch_size, self.transformer_dim, original_h, original_w
        )
        
        # 特征解码
        decoded_features = self.decoder(lidar_fused)
        
        # 感知任务推理
        perception_outputs = self.perception_heads(decoded_features)
        
        # 添加环境标记到输出
        perception_outputs['env_tokens'] = env_tokens
        
        return perception_outputs


def build_perception_network(config=None):
    """构建感知网络"""
    model = PerceptionNetwork(config)
    return model 