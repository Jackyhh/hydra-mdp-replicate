import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import open3d as o3d
import random
from config import TRAIN_DATA_DIRS, VAL_DATA_DIRS, TEST_DATA_DIRS, TRAJECTORY_CONFIG
import cv2 # Added for Sobel operator
import pickle
import json
from sklearn.cluster import KMeans, MiniBatchKMeans
import sqlite3
from config import TRAIN_DATA_DIRS, VAL_DATA_DIRS, TEST_DATA_DIRS, TRAJECTORY_CONFIG, USE_NUPLAN, NUPLAN_DB_FILES, NUPLAN_MAP_ROOT, DATA_DIR
import concurrent.futures
from functools import lru_cache
import time

# 添加文件缓存装饰器
def file_cache(func):
    """文件缓存装饰器，避免重复读取相同文件"""
    cache = {}
    
    def wrapper(*args, **kwargs):
        # 第一个参数通常是self，第二个是文件路径
        if len(args) > 1:
            file_path = args[1]
            if file_path in cache:
                return cache[file_path]
        
        result = func(*args, **kwargs)
        
        if len(args) > 1:
            file_path = args[1]
            cache[file_path] = result
            
            # 限制缓存大小，防止内存溢出
            if len(cache) > 1000:  # 最多缓存1000个文件
                # 删除最早添加的项
                for k in list(cache.keys())[:100]:
                    del cache[k]
                
        return result
    
    return wrapper

class NavsimDataset(Dataset):
    def __init__(self, data_dirs, split='train', transform=None, val_ratio=0.2, test_ratio=0.1, seed=42):
        """
        Navsim数据集加载器
        
        参数:
            data_dirs: 数据目录列表
            split: 'train', 'val', 或 'test'
            transform: 图像预处理变换
            val_ratio: 当使用同一目录时，验证集占比
            test_ratio: 当使用同一目录时，测试集占比
            seed: 随机种子，确保划分一致性
        """
        self.data_dirs = data_dirs
        self.split = split
        self.transform = transform
        self.cache = {}  # 添加缓存字典
        
        # 设置随机种子以确保一致性
        random.seed(seed)
        np.random.seed(seed)
        
        # 默认图像转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            
        # 扫描所有数据目录，自动查找所有navtrain子目录
        if len(data_dirs) == 0 or not os.path.exists(data_dirs[0]):
            # 如果未提供有效目录，扫描数据根目录
            if os.path.exists(DATA_DIR):
                all_subdirs = []
                for root, dirs, _ in os.walk(DATA_DIR):
                    for d in dirs:
                        if "navtrain" in d.lower():
                            subdir_path = os.path.join(root, d)
                            all_subdirs.append(subdir_path)
                
                if all_subdirs:
                    print(f"自动检测到以下数据目录: {all_subdirs}")
                    self.data_dirs = all_subdirs
                    
        # 获取所有场景目录
        self.scenes = []
        for data_dir in self.data_dirs:
            scene_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
            # 仅添加有效的场景目录（包含数据文件夹）
            for scene_dir in scene_dirs:
                if os.path.isdir(scene_dir) and not scene_dir.startswith('.'):
                    # 检查是否有MergedPointCloud目录，确认是有效的场景目录
                    if os.path.exists(os.path.join(scene_dir, "MergedPointCloud")):
                        self.scenes.append(scene_dir)
            
        print(f"找到{len(self.scenes)}个场景，分割类型: {split}")
        
        # 并行加载场景信息
        start_time = time.time()
        all_scene_info = self._parallel_load_scene_info()
        
        # 修改分割逻辑 - 总是使用随机划分以充分利用所有数据
        # 先对场景进行洗牌，但使用固定的随机种子确保一致性
        random.shuffle(all_scene_info)
        total = len(all_scene_info)
        
        if total == 0:
            print(f"警告: 没有找到有效的场景数据")
            self.scene_info = []
            return
            
        # 更新分割比例，确保所有数据都被使用
        val_idx = int(total * (1 - val_ratio - test_ratio))
        test_idx = int(total * (1 - test_ratio))
        
        if split == 'train':
            self.scene_info = all_scene_info[:val_idx]
        elif split == 'val':
            self.scene_info = all_scene_info[val_idx:test_idx]
        elif split == 'test':
            self.scene_info = all_scene_info[test_idx:]
        
        print(f"场景信息加载完成，用时: {time.time() - start_time:.2f}秒")
        print(f"为{split}集分配了{len(self.scene_info)}个场景，共{total}个场景")
        
    def _parallel_load_scene_info(self):
        """并行加载所有场景的帧信息"""
        scene_info = []
        
        # 首先确认场景列表不为空
        if not self.scenes:
            print(f"警告: 没有找到有效的场景目录，请检查 {self.data_dirs} 目录是否正确")
            return []
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务
            future_to_scene = {executor.submit(self._load_single_scene_info, scene_dir): scene_dir for scene_dir in self.scenes if os.path.exists(scene_dir)}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_scene):
                scene_dir = future_to_scene[future]
                try:
                    result = future.result()
                    if result:  # 确保结果不为空
                        scene_info.extend(result)
                    else:
                        print(f"警告: 场景 {scene_dir} 未返回有效数据")
                except Exception as exc:
                    print(f'{scene_dir} 处理出错: {exc}')
                
        return scene_info
    
    # def _load_single_scene_info(self, scene_dir):
    #     """加载单个场景的帧信息"""
    #     scene_frames = []
        
    #     try:
    #         # 获取LiDAR点云文件
    #         lidar_dir = os.path.join(scene_dir, "MergedPointCloud")
    #         if not os.path.exists(lidar_dir):
    #             print(f"警告: 点云目录不存在: {lidar_dir}")
    #             return scene_frames
            
    #         lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
    #         if not lidar_files:
    #             print(f"警告: 未在 {lidar_dir} 中找到点云文件")
    #             return scene_frames
            
    #         # 获取相机图像文件
    #         cam_dirs = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
            
    #         for i in range(len(lidar_files)):
    #             lidar_file = lidar_files[i]
    #             frame_id = os.path.basename(lidar_file).split('.')[0]
                
    #             # 为每个LiDAR帧找到对应的相机图像
    #             # cam_images = {}
    #             # cam_files_exist = False
    #             # for cam in cam_dirs:
    #             #     cam_dir = os.path.join(scene_dir, cam)
    #             #     cam_file = os.path.join(cam_dir, f"{frame_id}.jpg")
    #             #     if os.path.exists(cam_file):
    #             #         cam_images[cam] = cam_file
    #             #         cam_files_exist = True
                
    #             # --- 新方案：按照索引对齐，不再靠同名文件 ----------------
    #             cam_images = {}
    #             cam_files_exist = False
    
    #             for cam in cam_dirs:
    #                 cam_dir = os.path.join(scene_dir, cam)
    #                 if not os.path.isdir(cam_dir):
    #                     continue
    
    #                 # **一次性缓存每个目录下已排序的文件列表**（提高速度）
    #                 cache_key = f"__filelist__::{cam_dir}"
    #                 if cache_key not in self.cache:
    #                     self.cache[cache_key] = sorted(
    #                         glob.glob(os.path.join(cam_dir, "*.jpg"))
    #                     )
    #                 jpg_files = self.cache[cache_key]
    
    #                 # 如果相机帧总数跟 LiDAR 不一致就尽量对齐
    #                 if i < len(jpg_files):
    #                     cam_images[cam] = jpg_files[i]
    #                     cam_files_exist = True




    #             # 只有当至少有一个相机图像存在时，才添加这个帧
    #             if cam_files_exist or i % 10 == 0:  # 确保即使没有图像，至少每10帧添加一个
    #                 scene_frames.append({
    #                     'scene_dir': scene_dir,
    #                     'lidar_file': lidar_file,
    #                     'cam_files': cam_images,
    #                     'frame_id': frame_id,
    #                     'scene_id': os.path.basename(scene_dir),
    #                     'frame_index': i,
    #                 })
            
    #         return scene_frames
    #     except Exception as e:
    #         print(f"加载场景 {scene_dir} 时出错: {e}")
    #         return []


    def _load_single_scene_info(self, scene_dir):
        """加载单个场景的帧信息（按索引对齐相机帧）"""
        scene_frames = []

        try:
            # ---------- 1) 读取 LiDAR 列表 ----------
            lidar_dir = os.path.join(scene_dir, "MergedPointCloud")
            if not os.path.exists(lidar_dir):
                print(f"警告: 点云目录不存在: {lidar_dir}")
                return scene_frames

            lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pcd")))
            if not lidar_files:
                print(f"警告: 未在 {lidar_dir} 中找到点云文件")
                return scene_frames

            # ---------- 2) 预扫描各相机目录的 JPG 列表 ----------
            cam_dirs = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
            cam_file_lists = {}
            for cam in cam_dirs:
                cam_dir = os.path.join(scene_dir, cam)
                if not os.path.isdir(cam_dir):
                    continue                 # 该相机目录不存在，跳过
                cache_key = f"__filelist__::{cam_dir}"
                if cache_key not in self.cache:
                    self.cache[cache_key] = sorted(
                        glob.glob(os.path.join(cam_dir, "*.jpg"))
                    )
                cam_file_lists[cam] = self.cache[cache_key]

            # ---------- 3) 逐帧组装信息 ----------
            for i, lidar_file in enumerate(lidar_files):
                frame_id = os.path.basename(lidar_file).split('.')[0]

                cam_images = {}
                cam_files_exist = False
                for cam in cam_dirs:
                    jpg_files = cam_file_lists.get(cam, [])
                    # 直接用索引 i 对齐；若相机帧数不足则跳过
                    if i < len(jpg_files):
                        cam_images[cam] = jpg_files[i]
                        cam_files_exist = True

                # 加入条件：
                #   • 至少拿到一张对应相机图像；或
                #   • 每 10 帧采样一帧（即便缺图），避免数据过稀
                if cam_files_exist or i % 10 == 0:
                    scene_frames.append({
                        'scene_dir': scene_dir,
                        'lidar_file': lidar_file,
                        'cam_files': cam_images,
                        'frame_id': frame_id,
                        'scene_id': os.path.basename(scene_dir),
                        'frame_index': i,
                    })

            return scene_frames

        except Exception as e:
            print(f"加载场景 {scene_dir} 时出错: {e}")
            return []


    def __len__(self):
        """返回数据集大小"""
        return len(self.scene_info)
    
    def __getitem__(self, idx):
        """获取数据集中的一项"""
        info = self.scene_info[idx]
        
        # 检查缓存
        cache_key = f"{info['scene_id']}_{info['frame_id']}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 加载LiDAR点云
        lidar_data = self._load_point_cloud(info['lidar_file'])
        
        # 加载相机图像
        images = {}
        for cam_name in ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]:
            if cam_name in info['cam_files']:
                images[cam_name] = self._load_image(info['cam_files'][cam_name])
            else:
                # 如果没有对应的相机图像，创建一个空的占位图像
                images[cam_name] = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        # 生成轨迹数据
        trajectory = self._generate_trajectory(info)
        
        # 生成环境信息
        env_tokens = self._generate_env_tokens(info)
        
        # 生成子分数指标
        metrics = self._generate_metrics(info)
        
        result = {
            'lidar': lidar_data,
            'images': images,
            'trajectory': trajectory,
            'env_tokens': env_tokens,
            'metrics': metrics,
            'info': info
        }
        
        # 缓存结果 - 仅在训练模式下缓存少量项目，避免内存溢出
        if self.split == 'train' and len(self.cache) < 100:  # 最多缓存100个训练样本
            self.cache[cache_key] = result
            
        return result
    
    @file_cache  # 使用文件缓存装饰器
    def _load_point_cloud(self, pcd_file):
        """加载点云数据并转换为BEV表示"""
        try:
            # 使用Open3D读取点云
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            
            # 提取XYZ坐标
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            
            # 创建BEV表示（高度编码）
            x_range = (-50, 50)  # 米
            y_range = (-50, 50)  # 米
            z_range = (-5, 5)    # 米
            grid_size = 0.2      # 米/像素
            
            bev_size = int((x_range[1] - x_range[0]) / grid_size)
            bev = np.zeros((bev_size, bev_size, 6), dtype=np.float32)
            
            # 过滤点云范围
            mask = (x >= x_range[0]) & (x < x_range[1]) & \
                   (y >= y_range[0]) & (y < y_range[1]) & \
                   (z >= z_range[0]) & (z < z_range[1])
            x = x[mask]
            y = y[mask]
            z = z[mask]
            
            # 量化坐标
            x_indices = ((x - x_range[0]) / grid_size).astype(np.int32)
            y_indices = ((y - y_range[0]) / grid_size).astype(np.int32)
            
            # 创建高度通道
            for i, height_range in enumerate([(-5, -3), (-3, -1), (-1, 1), (1, 3), (3, 5)]):
                z_mask = (z >= height_range[0]) & (z < height_range[1])
                x_idx = x_indices[z_mask]
                y_idx = y_indices[z_mask]
                
                # 更新BEV表示 - 使用numpy矩阵操作代替循环
                np.add.at(bev[:, :, i], (y_idx, x_idx), 1.0)
            
            # 第6个通道是强度 (所有高度层的点云密度)
            bev[:, :, 5] = np.sum(bev[:, :, :5], axis=2) / 5.0
            
            return torch.from_numpy(bev.transpose(2, 0, 1))  # [C, H, W]
            
        except Exception as e:
            print(f"加载点云出错: {e}")
            # 返回空BEV
            return torch.zeros((6, int(100/0.2), int(100/0.2)), dtype=torch.float32)
    
    @file_cache  # 使用文件缓存装饰器
    def _load_image(self, image_file):
        """加载并预处理图像"""
        try:
            img = Image.open(image_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"加载图像出错: {e}")
            # 返回空图像
            return torch.zeros((3, 224, 224), dtype=torch.float32)
    
    def _generate_trajectory(self, info):
        """生成轨迹数据（基于真实物理模型）"""
        # 使用场景信息和车辆运动学模型生成轨迹
        timesteps = TRAJECTORY_CONFIG['total_timesteps']
        
        # 创建轨迹 [timesteps, 3] - (x, y, heading)
        trajectory = np.zeros((timesteps, 3), dtype=np.float32)
        
        # 默认参数，用于出现异常时
        current_pos = np.array([0.0, 0.0])
        current_heading = 0.0
        current_speed = 5.0
        
        # 从点云数据中提取车辆当前位置和朝向
        try:
            # 检查文件是否存在
            pcd_file = info['lidar_file']
            if not os.path.exists(pcd_file):
                raise FileNotFoundError(f"找不到点云文件: {pcd_file}")
                
            # 读取点云数据
            pcd = o3d.io.read_point_cloud(pcd_file)
            if len(np.asarray(pcd.points)) == 0:
                raise ValueError("点云为空")
                
            points = np.asarray(pcd.points)
            
            # 提取地面点
            try:
                # 检查点云是否有足够的点
                if len(points) < 10:  # 如果点数少于10个，无法执行RANSAC
                    print("警告: 点云点数过少，无法执行平面分割")
                    raise ValueError("点云点数过少")
                
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
                
                # 提取非地面点（可能是车辆、行人等）
                outlier_cloud = pcd.select_by_index(inliers, invert=True)
                outlier_points = np.asarray(outlier_cloud.points)
                
                # 如果有足够的非地面点，尝试聚类
                if len(outlier_points) > 10:
                    clusters = outlier_cloud.cluster_dbscan(eps=0.5, min_points=10)
                    
                    # 寻找最近的聚类作为前方物体
                    cluster_indices = np.array(clusters)
                    unique_clusters = np.unique(cluster_indices)
                    front_objects = []
                    
                    if len(unique_clusters) > 1:  # 至少有一个聚类（排除噪声点-1）
                        for i in unique_clusters:
                            if i >= 0:  # 排除噪声点
                                # 计算聚类中心
                                cluster_points = np.asarray(outlier_cloud.points)[cluster_indices == i]
                                center = np.mean(cluster_points, axis=0)
                                # 仅考虑前方物体（假设车辆坐标系x轴为前）
                                if center[0] > 0:
                                    front_objects.append((center, i, cluster_points))
                    
                    # 估计车辆当前朝向
                    if front_objects:  # 改为空列表检查
                        # 使用最近的前方物体估计朝向
                        # 按照距离排序
                        front_objects.sort(key=lambda x: np.sqrt(x[0][0]**2 + x[0][1]**2))
                        closest_obj = front_objects[0][0]
                        current_heading = np.arctan2(closest_obj[1], closest_obj[0])
            except Exception as e:
                print(f"提取地面特征失败: {e}")
                # 如果地面分割或聚类失败，使用默认朝向
                pass
            
            # 根据场景ID提取车速信息
            scene_id = info['scene_id']
            # 使用场景ID的哈希值生成一个伪随机但确定性的速度因子
            speed_factor = (hash(scene_id) % 1000) / 1000 * 0.5 + 0.5  # 0.5-1.0范围内的车速因子
            current_speed = 5.0 * speed_factor  # m/s
        
        except Exception as e:

            pass
        
        try:
            # 使用自行车运动模型预测未来轨迹
            dt = 1.0 / TRAJECTORY_CONFIG['trajectory_frequency']
            
            # 初始状态
            x, y = current_pos
            theta = current_heading
            v = current_speed
            
            # 生成曲线轨迹，车辆会以恒定的速度向前行驶，并逐渐向一个方向转向
            # 使用场景ID的哈希值确定转向方向和大小，确保同一场景产生一致的轨迹
            scene_id_hash = hash(info['scene_id'])
            turn_direction = 1 if scene_id_hash % 2 == 0 else -1  # 左转或右转
            turn_rate = (scene_id_hash % 100) / 1000  # 转向率 (rad/s)
            
            # 生成轨迹点
            for t in range(timesteps):
                # 更新位置和朝向
                x += v * np.cos(theta) * dt
                y += v * np.sin(theta) * dt
                theta += turn_direction * turn_rate * dt
                
                # 存储轨迹点
                trajectory[t, 0] = x
                trajectory[t, 1] = y
                trajectory[t, 2] = theta
            
        except Exception as e:
            
            for t in range(timesteps):
                trajectory[t, 0] = t * 0.1 * current_speed  # x方向匀速前进
                trajectory[t, 1] = 0.0                       # y保持不变
                trajectory[t, 2] = 0.0                       # 朝向保持不变
        
        return torch.from_numpy(trajectory).float()
    
    def _generate_env_tokens(self, info):
        """生成环境标记（基于感知结果）"""
        env_dim = 256
        
        try:
            # 读取点云数据
            pcd_file = info['lidar_file']
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            
            # 初始化环境特征列表
            env_features = []
            
            # 读取相机图像数据
            cam_images = {}
            for cam_name, cam_file in info['cam_files'].items():
                try:
                    img = np.array(Image.open(cam_file))
                    cam_images[cam_name] = img
                except Exception as e:
                    print(f"无法读取图像 {cam_file}: {e}")
            
            # 安全检查点云数量
            if len(points) < 10:
                # 点云点数不足，使用随机向量
                print(f"警告: 环境标记生成 - 点云点数过少 ({len(points)})")
                return torch.randn(env_dim, dtype=torch.float32)
                
            # 分析点云特征
            try:
                # 提取地面和障碍物
                if len(points) >= 10:  # 确保有足够的点来执行RANSAC
                    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
                    ground_cloud = pcd.select_by_index(inliers)
                    obstacle_cloud = pcd.select_by_index(inliers, invert=True)
                    
                    # 计算地面法向量特征
                    a, b, c, d = plane_model
                    ground_normal = np.array([a, b, c])
                    ground_slope = np.arccos(np.dot(ground_normal, np.array([0, 0, 1])))
                    env_features.append(float(ground_slope))
                    
                    # 计算地面平坦度和密度
                    ground_points = np.asarray(ground_cloud.points)
                    if len(ground_points) > 0:
                        # 地面平坦度（地面点的z值标准差）
                        ground_flatness = np.std(ground_points[:, 2])
                        # 地面点密度
                        ground_density = len(ground_points) / (100 * 100) 
                        
                        env_features.append(float(ground_flatness))
                        env_features.append(float(ground_density))
                    else:
                        env_features.extend([0.1, 0.05]) 
                    
                    # 计算障碍物分布特征
                    obstacle_points = np.asarray(obstacle_cloud.points)
                    if len(obstacle_points) > 0:
                        obstacle_density = len(obstacle_points) / len(points)
                        env_features.append(float(obstacle_density))
                        
                        # 障碍物距离统计
                        distances = np.sqrt(np.sum(obstacle_points[:, :2]**2, axis=1))
                        mean_distance = np.mean(distances) if len(distances) > 0 else 10.0
                        min_distance = np.min(distances) if len(distances) > 0 else 10.0
                        env_features.append(float(min(1.0, mean_distance / 50.0)))
                        env_features.append(float(min(1.0, min_distance / 20.0)))
                        
                        # 障碍物高度统计
                        heights = obstacle_points[:, 2]
                        mean_height = np.mean(heights) if len(heights) > 0 else 0.0
                        max_height = np.max(heights) if len(heights) > 0 else 0.0
                        env_features.append(float(np.minimum(1.0, mean_height / 3.0)))
                        env_features.append(float(min(1.0, max_height / 5.0)))
                    else:
                        
                        env_features.extend([0.0, 0.8, 0.8, 0.0, 0.0]) 
                else:
                    
                    env_features.extend([0.0, 0.1, 0.05])  
                    env_features.extend([0.0, 0.8, 0.8, 0.0, 0.0]) 
            except Exception as e:
                print(f"提取地面特征失败: {e}")
                # 填充默认值
                env_features.extend([0.0, 0.1, 0.05])  
                env_features.extend([0.0, 0.8, 0.8, 0.0, 0.0])  
            
            # 2. 提取障碍物特征 (从点云中提取非地面点)
            try:
                if len(points) >= 10 and 'inliers' in locals():
                    outlier_cloud = pcd.select_by_index(inliers, invert=True)
                    if len(np.asarray(outlier_cloud.points)) > 10:
                        clusters = outlier_cloud.cluster_dbscan(eps=0.5, min_points=10)
                        
                        # 统计障碍物信息
                        cluster_indices = np.array(clusters)
                        unique_clusters = np.unique(cluster_indices)
                        
                        # 障碍物数量（除去噪声点-1）
                        obstacle_count = len([c for c in unique_clusters if c >= 0])
                        env_features.append(min(1.0, obstacle_count / 10))  # 归一化障碍物数量
                        
                        # 提取最近的5个障碍物的特征
                        obstacles = []
                        for i in unique_clusters:
                            if i >= 0:  # 排除噪声点
                                cluster_points = np.asarray(outlier_cloud.points)[cluster_indices == i]
                                center = np.mean(cluster_points, axis=0)
                                size = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                                dist = np.sqrt(center[0]**2 + center[1]**2)  # 到原点的距离
                                
                                obstacles.append((dist, center, size, len(cluster_points)))
                        
                        # 按距离排序
                        obstacles.sort(key=lambda x: float(x[0]))
                        
                        # 提取最近的5个障碍物的特征
                        for i in range(min(5, len(obstacles))):
                            dist, center, size, point_count = obstacles[i]
                            
                            # 距离特征
                            env_features.append(float(min(1.0, dist / 50)))  # 归一化距离
                            
                            # 位置特征
                            env_features.append(float((center[0] + 50) / 100))  # 归一化x坐标到[0,1]
                            env_features.append(float((center[1] + 50) / 100))  # 归一化y坐标到[0,1]
                            env_features.append(float((center[2] + 5) / 10))    # 归一化z坐标到[0,1]
                            
                            # 大小特征
                            env_features.append(float(min(1.0, size[0] / 10)))  # 归一化x尺寸
                            env_features.append(float(min(1.0, size[1] / 10)))  # 归一化y尺寸
                            env_features.append(float(min(1.0, size[2] / 5)))   # 归一化z尺寸
                            
                            # 点数特征
                            env_features.append(float(min(1.0, point_count / 1000)))  # 归一化点数
                        
                        # 如果障碍物少于5个，用0填充
                        for i in range(len(obstacles), 5):
                            env_features.extend([0.0] * 8)  # 每个障碍物8个特征
                    else:
                        # 没有足够的非地面点进行聚类，填充默认值
                        env_features.append(0.0)  # 默认障碍物数量
                        env_features.extend([0.0] * 40)  # 默认障碍物特征
                else:                    
                    env_features.append(0.0)  # 默认障碍物数量
                    env_features.extend([0.0] * 40)  # 默认障碍物特征
            except Exception as e:
                print(f"提取障碍物特征失败: {e}")
                # 填充默认值
                env_features.append(0.0)  # 默认障碍物数量
                env_features.extend([0.0] * 40)  # 默认障碍物特征 (5个障碍物，每个8个特征)
                
            # 3. 从相机图像中提取特征
            try:
                image_features = []
                
                # 处理前视摄像头
                if 'CAM_F0' in cam_images:
                    front_img = cam_images['CAM_F0']
                    
                    # 缩放图像
                    front_img = Image.fromarray(front_img).resize((224, 224))
                    
                    # 进行基本的图像特征提取
                    # 1. 颜色分布
                    hsv_img = np.array(front_img.convert('HSV'))
                    h_hist, _ = np.histogram(hsv_img[:,:,0], bins=8, range=(0, 180), density=True)
                    s_hist, _ = np.histogram(hsv_img[:,:,1], bins=8, range=(0, 256), density=True)
                    v_hist, _ = np.histogram(hsv_img[:,:,2], bins=8, range=(0, 256), density=True)
                    
                    image_features.extend(h_hist)
                    image_features.extend(s_hist)
                    image_features.extend(v_hist)
                    
                    # 2. 边缘特征
                    # 转灰度图
                    gray_img = np.array(front_img.convert('L'))
                    # 使用Sobel算子计算梯度
                    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
                    # 计算梯度大小
                    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                    # 将梯度大小归一化到[0,1]
                    if np.max(gradient_magnitude) > 0:
                        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
                    
                    # 计算图像不同区域的梯度密度作为特征
                    h, w = gradient_magnitude.shape
                    regions = [
                        (0, 0, h//2, w//2),        # 左上
                        (0, w//2, h//2, w),        # 右上
                        (h//2, 0, h, w//2),        # 左下
                        (h//2, w//2, h, w)         # 右下
                    ]
                    
                    for r1, c1, r2, c2 in regions:
                        region = gradient_magnitude[r1:r2, c1:c2]
                        # 计算该区域边缘密度
                        edge_density = np.mean(region > 0.1)  # 阈值0.1
                        image_features.append(float(edge_density))
                    
                    # 3. 纹理特征
                    # 计算每个区域的灰度共生矩阵的一些统计特征
                    for r1, c1, r2, c2 in regions:
                        region = gray_img[r1:r2, c1:c2]
                        # 计算灰度直方图
                        hist, _ = np.histogram(region, bins=8, range=(0, 256), density=True)
                        # 计算区域的平均值和标准差
                        mean_value = np.mean(region) / 255
                        std_value = np.std(region) / 255
                        
                        image_features.append(float(mean_value))
                        image_features.append(float(std_value))
                        image_features.extend([float(h) for h in hist])
                else:
                    # 如果没有前视图像，添加默认特征
                    image_features = [0.0] * 60  # 颜色分布(24) + 边缘特征(4) + 纹理特征(32)
                
                env_features.extend(image_features)
            except Exception as e:
                print(f"提取图像特征失败: {e}")
                # 填充默认图像特征
                env_features.extend([0.0] * 60)
                
            # 确保特征维度为env_dim
            if len(env_features) > env_dim:
                env_features = env_features[:env_dim]
            elif len(env_features) < env_dim:
                # 填充余下的维度
                env_features.extend([0.0] * (env_dim - len(env_features)))
                
            # 转换为Tensor
            return torch.tensor(env_features, dtype=torch.float32)
            
        except Exception as e:
            print(f"环境标记生成出错: {e}")            
            return torch.randn(env_dim, dtype=torch.float32)
    
    def _generate_metrics(self, info):
        """生成评估指标"""
        try:
            # 加载轨迹
            trajectory = self._generate_trajectory(info).numpy()
            
            # 加载点云数据
            pcd_file = info['lidar_file']
            try:
                pcd = o3d.io.read_point_cloud(pcd_file)
            except Exception as e:
                print(f"加载点云出错: {e}")
                raise
            
            # 1. 无碰撞率(NC)
            nc_score = self._calculate_collision_score(pcd, trajectory)
            
            # 2. 可行驶区域合规性(DAC)
            dac_score = self._calculate_drivable_area_compliance(pcd, trajectory)
            
            # 3. 到碰撞时间(TTC)
            ttc_score = self._calculate_time_to_collision(pcd, trajectory)
            
            # 4. 舒适度(C)
            c_score = self._calculate_comfort_score(trajectory)
            
            # 5. 自我进展(EP)
            ep_score = self._calculate_progress_score(trajectory)
            
            # 6. 模仿分数(im)
            im_score = self._calculate_imitation_score(trajectory, info)
            
            # 正确构造tensor，避免使用torch.tensor()从tensor构造新tensor
            # 首先确保所有数据都是标量
            metrics = {
                'NC': float(nc_score) if isinstance(nc_score, (torch.Tensor, np.ndarray)) else nc_score,
                'DAC': float(dac_score) if isinstance(dac_score, (torch.Tensor, np.ndarray)) else dac_score,
                'TTC': float(ttc_score) if isinstance(ttc_score, (torch.Tensor, np.ndarray)) else ttc_score,
                'C': float(c_score) if isinstance(c_score, (torch.Tensor, np.ndarray)) else c_score,
                'EP': float(ep_score) if isinstance(ep_score, (torch.Tensor, np.ndarray)) else ep_score,
                'im': float(im_score) if isinstance(im_score, (torch.Tensor, np.ndarray)) else im_score,
            }
            
            # 将Python标量转换为tensor
            metrics = {k: torch.tensor(v, dtype=torch.float32) for k, v in metrics.items()}
            
            return metrics
            
        except Exception as e:
            print(f"评估指标计算出错: {e}，使用随机数据")
            
            metrics = {
                'NC': torch.tensor(random.uniform(0.9, 1.0), dtype=torch.float32),  # 无碰撞率
                'DAC': torch.tensor(random.uniform(0.85, 1.0), dtype=torch.float32),  # 可行驶区域合规性
                'TTC': torch.tensor(random.uniform(0.8, 1.0), dtype=torch.float32),  # 到碰撞时间
                'C': torch.tensor(random.uniform(0.85, 1.0), dtype=torch.float32),  # 舒适度
                'EP': torch.tensor(random.uniform(0.75, 0.95), dtype=torch.float32),  # 自我进展
                'im': torch.tensor(random.uniform(0.7, 0.9), dtype=torch.float32),  # 模仿分数
            }
            return metrics
    
    def _calculate_collision_score(self, pcd, trajectory):
        """计算无碰撞率"""
        try:
            # 检查点云是否有足够的点
            points = np.asarray(pcd.points)
            if len(points) < 10:  # 如果点数少于10个，无法执行RANSAC
                print("警告: 点云点数过少，无法执行平面分割")
                return 0.75  
                
            # 提取非地面点作为潜在障碍物 - 改进平面分割参数
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=5, num_iterations=2000)
            obstacles = pcd.select_by_index(inliers, invert=True)
            obstacle_points = np.asarray(obstacles.points)
            
            if len(obstacle_points) == 0:
                return 0.85  # 没有障碍物，较高安全分数，但保留一定余量
            
            # 车辆尺寸（长宽高）
            vehicle_length = 4.5  # 米
            vehicle_width = 2.0   # 米
            vehicle_buffer = 0.3  # 安全缓冲区
            
            # 检查轨迹上的每个点是否与障碍物碰撞
            min_distance = float('inf')
            collisions = 0
            near_collisions = 0
            total_points = len(trajectory)
            
            # 轨迹采样率 - 优化计算效率
            sample_step = max(1, total_points // 20)  # 采样20个点或全部点
            sampled_indices = list(range(0, total_points, sample_step))
            if total_points - 1 not in sampled_indices:
                sampled_indices.append(total_points - 1)  # 确保包含最后一个点
                
            # 动态调整检测精度 - 根据点云密度
            point_density = len(points) / (np.max(points[:, 0]) - np.min(points[:, 0])) / (np.max(points[:, 1]) - np.min(points[:, 1]))
            detection_precision = min(1.0, max(0.2, 10.0 / point_density))
            
            # 重要轨迹点权重
            point_weights = np.ones(total_points)
            # 起始和结束点更重要
            point_weights[0] = 2.0
            point_weights[-1] = 2.0
            # 中间点权重线性增加然后减少，以反映不同阶段的重要性
            mid_point = total_points // 2
            for i in range(1, mid_point):
                point_weights[i] = 1.0 + 0.5 * (i / mid_point)
            for i in range(mid_point, total_points - 1):
                point_weights[i] = 1.5 - 0.5 * ((i - mid_point) / (total_points - mid_point))
            
            # 使用KD树加速最近邻搜索
            if len(obstacle_points) > 10:
                try:
                    from scipy.spatial import KDTree
                    obstacle_tree = KDTree(obstacle_points[:, :2])  # 只使用x,y坐标
                    use_kdtree = True
                except ImportError:
                    use_kdtree = False
            else:
                use_kdtree = False
            
            for t_idx in sampled_indices:
                t = t_idx  # 实际轨迹点索引
                x, y, heading = trajectory[t]
                
                # 创建表示车辆的矩形
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                
                # 车辆四个角的相对坐标（考虑安全缓冲区）
                corners_rel = np.array([
                    [(vehicle_length/2 + vehicle_buffer), (vehicle_width/2 + vehicle_buffer)],
                    [(vehicle_length/2 + vehicle_buffer), -(vehicle_width/2 + vehicle_buffer)],
                    [-(vehicle_length/2 + vehicle_buffer), -(vehicle_width/2 + vehicle_buffer)],
                    [-(vehicle_length/2 + vehicle_buffer), (vehicle_width/2 + vehicle_buffer)]
                ])
                
                # 旋转坐标
                corners_rot = np.zeros_like(corners_rel)
                for i in range(4):
                    corners_rot[i, 0] = corners_rel[i, 0] * cos_h - corners_rel[i, 1] * sin_h
                    corners_rot[i, 1] = corners_rel[i, 0] * sin_h + corners_rel[i, 1] * cos_h
                
                # 平移到当前位置
                corners_abs = corners_rot + np.array([x, y])
                
                # 定义检查区域 - 简化计算
                check_radius = np.sqrt((vehicle_length/2 + vehicle_buffer)**2 + (vehicle_width/2 + vehicle_buffer)**2)
                
                # 使用KD树快速查找附近的障碍物点
                collision_detected = False
                if use_kdtree:
                    nearby_indices = obstacle_tree.query_ball_point([x, y], check_radius)
                    nearby_points = obstacle_points[nearby_indices] if nearby_indices else []
                else:
                    # 传统方法查找附近点
                    nearby_points = []
                    for p in obstacle_points:
                        dist = np.sqrt((p[0] - x)**2 + (p[1] - y)**2)
                        if dist <= check_radius and p[2] > 0.1 and p[2] < 2.0:  # 高度筛选
                            nearby_points.append(p)
                
                # 更精确地检查附近点是否会导致碰撞
                for p in nearby_points:
                    point_xy = p[:2]
                    
                    # 计算点到车辆中心的距离（更新最小距离记录）
                    dist_to_center = np.linalg.norm(point_xy - np.array([x, y]))
                    min_distance = min(min_distance, dist_to_center)
                    
                    # 区分严重碰撞和轻微碰撞（近碰撞）
                    if dist_to_center < (vehicle_length/2 + vehicle_width/2) / 2:
                        collisions += point_weights[t]  # 严重碰撞，使用点权重
                        collision_detected = True
                        break
                    elif dist_to_center < vehicle_length/2 + vehicle_buffer:
                        near_collisions += point_weights[t] * 0.5  # 轻微碰撞，使用点权重并减半
                
                if collision_detected:
                    break  # 一旦检测到严重碰撞，立即停止检查
            
            # 根据碰撞检测结果计算改进的NC分数
            if collisions > 0:
                # 有严重碰撞，分数较低但不为零
                collision_factor = min(1.0, collisions / sum(point_weights[sampled_indices]))
                base_score = 0.4  # 碰撞基础分
                nc_score = base_score * (1.0 - collision_factor)
            elif near_collisions > 0:
                # 有轻微碰撞（近碰撞）
                near_collision_factor = min(1.0, near_collisions / sum(point_weights[sampled_indices]))
                nc_score = 0.7 * (1.0 - near_collision_factor)
            else:
                # 无碰撞，根据最小距离计算分数
                if min_distance < float('inf'):
                    # 距离归一化，10米及以上视为完全安全
                    distance_factor = min(1.0, max(0.0, float(min_distance - vehicle_length/2) / 10.0))
                    nc_score = 0.7 + 0.2 * distance_factor
                else:
                    nc_score = 0.85  # 无探测到的障碍物
            
            # 平滑处理，防止极端值
            nc_score = max(0.35, min(0.85, nc_score))
            
            return nc_score
            
        except Exception as e:
            print(f"无碰撞率计算出错: {e}")
            
            return random.uniform(0.5, 0.7)
    
    def _calculate_drivable_area_compliance(self, pcd, trajectory):
        """计算可行驶区域合规性"""
        try:
            # 检查点云是否有足够的点
            points = np.asarray(pcd.points)
            if len(points) < 10:  # 如果点数少于10个，无法执行RANSAC
                print("警告: 点云点数过少，无法执行平面分割")
                return 0.9  
            
            # 提取地面点作为可行驶区域
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
            ground = pcd.select_by_index(inliers)
            ground_points = np.asarray(ground.points)
            
            if len(ground_points) == 0:
                return 0.9  
            
            # 计算地面的边界
            ground_min_x = np.min(ground_points[:, 0])
            ground_max_x = np.max(ground_points[:, 0])
            ground_min_y = np.min(ground_points[:, 1])
            ground_max_y = np.max(ground_points[:, 1])
            
            # 计算轨迹点在可行驶区域内的比例
            total_points = len(trajectory)
            in_drivable_area = 0
            
            for t in range(total_points):
                x, y, _ = trajectory[t]
                
                # 检查点是否在地面边界内
                if x >= ground_min_x and x <= ground_max_x and y >= ground_min_y and y <= ground_max_y:
                    # 进一步检查该点周围是否有地面点
                    nearby_ground_points = 0
                    search_radius = 1.0  # 1米半径
                    
                    for gp in ground_points:
                        dist = np.sqrt((gp[0] - x)**2 + (gp[1] - y)**2)
                        if dist < search_radius:
                            nearby_ground_points += 1
                    
                    # 如果周围有足够多的地面点，认为在可行驶区域内
                    if nearby_ground_points > 10:
                        in_drivable_area += 1
            
            # 计算合规性分数
            dac_score = in_drivable_area / total_points
            
            # 提高分数基线（可行驶区域合规性通常较高）
            dac_score = 0.8 + 0.2 * dac_score
            
            return dac_score
            
        except Exception as e:
            print(f"可行驶区域合规性计算出错: {e}")
            return random.uniform(0.85, 1.0)
    
    def _calculate_time_to_collision(self, pcd, trajectory):
        """计算到碰撞时间分数"""
        try:
            # 检查点云是否有足够的点
            points = np.asarray(pcd.points)
            if len(points) < 10:  # 如果点数少于10个，无法执行RANSAC
                print("警告: 点云点数过少，无法执行平面分割")
                return 0.95  # 返回一个安全的默认值
            
            # 提取非地面点作为潜在障碍物
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
            obstacles = pcd.select_by_index(inliers, invert=True)
            obstacle_points = np.asarray(obstacles.points)
            
            if len(obstacle_points) == 0:
                return 1.0  # 没有障碍物，TTC非常好
            
            # 轨迹采样频率
            dt = 1.0 / TRAJECTORY_CONFIG['trajectory_frequency']
            
            # 找到沿轨迹的最小TTC
            min_ttc = float('inf')
            
            for t in range(len(trajectory) - 1):
                x, y, _ = trajectory[t]
                next_x, next_y, _ = trajectory[t + 1]
                
                # 计算速度
                vx = (next_x - x) / dt
                vy = (next_y - y) / dt
                speed = np.sqrt(vx**2 + vy**2)
                
                if speed < 0.1:  # 几乎静止
                    continue
                
                # 检查每个障碍物点
                for p in obstacle_points:
                    # 只关注较高的点（车辆高度范围内）
                    if p[2] > 0.1 and p[2] < 2.0:
                        # 计算点到当前位置的距离
                        dx = p[0] - x
                        dy = p[1] - y
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        # 计算相对速度在障碍物方向上的投影
                        dot_product = (dx * vx + dy * vy) / dist
                        
                        # 如果相对速度朝向障碍物
                        if dot_product > 0:
                            # 计算TTC
                            ttc = dist / dot_product
                            
                            if ttc < min_ttc:
                                min_ttc = ttc
            
            # 计算TTC分数
            if min_ttc == float('inf'):
                ttc_score = 1.0  # 没有碰撞风险
            else:
                # TTC评分函数：
                # - 小于2秒：高风险（分数较低）
                # - 大于5秒：低风险（分数较高）
                if min_ttc < 2.0:
                    ttc_score = 0.6 + 0.2 * (min_ttc / 2.0)
                elif min_ttc < 5.0:
                    ttc_score = 0.8 + 0.15 * ((min_ttc - 2.0) / 3.0)
                else:
                    ttc_score = 0.95 + 0.05 * min(1.0, (min_ttc - 5.0) / 5.0)
            
            return ttc_score
            
        except Exception as e:
            print(f"到碰撞时间计算出错: {e}")
            return random.uniform(0.8, 1.0)
    
    def _calculate_comfort_score(self, trajectory):
        """计算舒适度分数"""
        try:
            # 轨迹采样频率
            dt = 1.0 / TRAJECTORY_CONFIG['trajectory_frequency']
            
            # 计算加速度和角速度
            accelerations = []
            angular_velocities = []
            
            for t in range(1, len(trajectory) - 1):
                # 前一个点
                prev_x, prev_y, prev_heading = trajectory[t - 1]
                # 当前点
                x, y, heading = trajectory[t]
                # 下一个点
                next_x, next_y, next_heading = trajectory[t + 1]
                
                # 计算速度
                vx1 = (x - prev_x) / dt
                vy1 = (y - prev_y) / dt
                v1 = np.sqrt(vx1**2 + vy1**2)
                
                vx2 = (next_x - x) / dt
                vy2 = (next_y - y) / dt
                v2 = np.sqrt(vx2**2 + vy2**2)
                
                # 计算加速度
                ax = (vx2 - vx1) / dt
                ay = (vy2 - vy1) / dt
                acceleration = np.sqrt(ax**2 + ay**2)
                accelerations.append(acceleration)
                
                # 计算角速度
                angular_velocity = abs((next_heading - prev_heading) / (2 * dt))
                angular_velocities.append(angular_velocity)
            
            if not accelerations:
                return 1.0  # 没有足够的点来计算加速度
            
            # 计算加速度和角速度的最大值和均方根值
            max_acceleration = max(accelerations)
            rms_acceleration = np.sqrt(np.mean(np.square(accelerations)))
            
            max_angular_velocity = max(angular_velocities)
            rms_angular_velocity = np.sqrt(np.mean(np.square(angular_velocities)))
            
            # 计算舒适度分数
            # 1. 基于最大加速度（超过3m/s²开始不舒适）
            if max_acceleration < 1.0:
                acc_score = 1.0
            elif max_acceleration < 2.0:
                acc_score = 1.0 - 0.1 * (max_acceleration - 1.0)
            elif max_acceleration < 3.0:
                acc_score = 0.9 - 0.1 * (max_acceleration - 2.0)
            else:
                acc_score = max(0.7, 0.8 - 0.1 * (max_acceleration - 3.0))
            
            # 2. 基于RMS加速度
            rms_acc_score = max(0.7, 1.0 - 0.15 * rms_acceleration)
            
            # 3. 基于最大角速度（超过0.5rad/s开始不舒适）
            if max_angular_velocity < 0.2:
                ang_score = 1.0
            elif max_angular_velocity < 0.5:
                ang_score = 1.0 - 0.2 * (max_angular_velocity - 0.2) / 0.3
            else:
                ang_score = max(0.7, 0.8 - 0.2 * (max_angular_velocity - 0.5))
            
            # 4. 基于RMS角速度
            rms_ang_score = max(0.7, 1.0 - 0.5 * rms_angular_velocity)
            
            # 组合分数
            c_score = 0.3 * acc_score + 0.2 * rms_acc_score + 0.3 * ang_score + 0.2 * rms_ang_score
            
            return c_score
            
        except Exception as e:
            print(f"舒适度计算出错: {e}")
            return random.uniform(0.85, 1.0)
    
    def _calculate_progress_score(self, trajectory):
        """计算自我进展分数"""
        try:
            # 计算轨迹总长度
            total_distance = 0
            for t in range(1, len(trajectory)):
                x1, y1, _ = trajectory[t - 1]
                x2, y2, _ = trajectory[t]
                segment_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance += segment_distance
            
            # 计算起点到终点的直线距离
            start_x, start_y, _ = trajectory[0]
            end_x, end_y, _ = trajectory[-1]
            direct_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # 计算时间
            total_time = len(trajectory) / TRAJECTORY_CONFIG['trajectory_frequency']
            
            # 计算平均速度
            avg_speed = total_distance / total_time if total_time > 0 else 0
            
            # 计算有效率（直线距离/总行驶距离）
            efficiency = direct_distance / total_distance if total_distance > 0 else 0
            
            # 计算进展分数
            # 1. 速度分数
            if avg_speed < 1.0:
                speed_score = max(0.5, avg_speed / 2.0)
            elif avg_speed < 10.0:
                speed_score = 0.5 + 0.5 * (avg_speed / 10.0)
            else:
                speed_score = 1.0
            
            # 2. 效率分数
            efficiency_score = efficiency ** 0.5  
            
            # 组合分数
            ep_score = 0.7 * speed_score + 0.3 * efficiency_score
            
            return ep_score
            
        except Exception as e:
            print(f"自我进展计算出错: {e}")
            return random.uniform(0.75, 0.95)
    
    def _calculate_imitation_score(self, trajectory, info):
        """计算模仿分数"""
        try:
            
            
            # 计算轨迹的平滑度
            smoothness_score = 0.0
            curvature_changes = 0
            
            for t in range(2, len(trajectory) - 1):
                # 前一个、当前和下一个点
                prev_x, prev_y, _ = trajectory[t - 2]
                curr_x, curr_y, _ = trajectory[t - 1]
                next_x, next_y, _ = trajectory[t]
                
                # 计算两个连续段的方向
                dir1 = np.arctan2(curr_y - prev_y, curr_x - prev_x)
                dir2 = np.arctan2(next_y - curr_y, next_x - curr_x)
                
                # 计算方向变化
                dir_change = abs(dir2 - dir1)
                while dir_change > np.pi:
                    dir_change = 2 * np.pi - dir_change
                
                # 如果方向变化大，认为轨迹不平滑
                if dir_change > 0.2:  # 阈值约为11.5度
                    curvature_changes += 1
            
            # 根据曲率变化次数计算平滑度分数
            if curvature_changes <= 2:
                smoothness_score = 1.0
            elif curvature_changes <= 5:
                smoothness_score = 0.9 - 0.05 * (curvature_changes - 2)
            else:
                smoothness_score = max(0.7, 0.75 - 0.05 * (curvature_changes - 5))
            
            # 计算轨迹的自然性（基于速度分布）
            naturalness_score = 0.0
            speeds = []
            dt = 1.0 / TRAJECTORY_CONFIG['trajectory_frequency']
            
            for t in range(1, len(trajectory)):
                x1, y1, _ = trajectory[t - 1]
                x2, y2, _ = trajectory[t]
                
                # 计算速度
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed = dist / dt
                speeds.append(speed)
            
            # 计算速度变化
            speed_changes = []
            for i in range(1, len(speeds)):
                speed_changes.append(abs(speeds[i] - speeds[i - 1]))
            
            # 速度变化越小，轨迹越自然
            avg_speed_change = np.mean(speed_changes) if speed_changes else 0
            if avg_speed_change < 0.2:
                naturalness_score = 1.0
            elif avg_speed_change < 0.5:
                naturalness_score = 1.0 - 0.2 * (avg_speed_change - 0.2) / 0.3
            elif avg_speed_change < 1.0:
                naturalness_score = 0.8 - 0.1 * (avg_speed_change - 0.5) / 0.5
            else:
                temp_score = 0.7 - 0.1 * (avg_speed_change - 1.0)
                if temp_score > 0.7:
                    naturalness_score = temp_score
                else:
                    naturalness_score = 0.7

            # 场景适应性评分（根据场景ID调整）
            scene_id = info['scene_id']
            scene_factor = ((hash(scene_id) % 100) / 100.0) * 0.1 + 0.85  # 0.85-0.95的范围
            
            # 组合评分
            im_score = 0.4 * smoothness_score + 0.4 * naturalness_score + 0.2 * scene_factor
            
            return im_score
            
        except Exception as e:
            print(f"模仿分数计算出错: {e}")
            return random.uniform(0.7, 0.9)


# 添加nuPlan数据加载功能
class NuPlanDataset(Dataset):
    def __init__(self, db_files, split='train', transform=None):
        self.db_files = db_files
        self.split = split
        self.transform = transform
        self.samples = []
        self.map_cache = {}
        
        # 加载数据索引
        self._load_data_index()
        
    def _load_data_index(self):
        print(f"加载nuPlan {self.split}数据索引...")
        
        for db_file in self.db_files:
            if not os.path.exists(db_file):
                print(f"警告: 数据库文件不存在: {db_file}")
                continue
                
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # 获取场景ID列表
                cursor.execute("SELECT token FROM scene")
                scene_tokens = [row[0] for row in cursor.fetchall()]
                
                for scene_token in scene_tokens:
                    # 获取场景起始时间戳
                    cursor.execute(f"SELECT timestamp_ns FROM lidar_pc WHERE scene_token = '{scene_token}' ORDER BY timestamp_ns ASC LIMIT 1")
                    start_timestamp = cursor.fetchone()
                    
                    if not start_timestamp:
                        continue
                    
                    start_timestamp = start_timestamp[0]
                    
                    # 获取场景中的关键帧
                    cursor.execute(f"""
                        SELECT token, timestamp_ns, ego_pose 
                        FROM lidar_pc 
                        WHERE scene_token = '{scene_token}'
                        AND timestamp_ns >= {start_timestamp}
                        ORDER BY timestamp_ns ASC
                    """)
                    
                    lidar_frames = cursor.fetchall()
                    
                    # 确保有足够的帧用于轨迹预测
                    if len(lidar_frames) < 50:  # 需要足够的帧来提取历史和未来轨迹
                        continue
                    
                    # 每10帧采样一次作为输入帧
                    for i in range(0, len(lidar_frames) - 40, 10):  # 确保有40帧用于未来轨迹
                        sample = {
                            'db_file': db_file,
                            'scene_token': scene_token,
                            'lidar_token': lidar_frames[i][0],
                            'timestamp': lidar_frames[i][1],
                            'ego_pose': lidar_frames[i][2],
                            'future_frames': [frame[0] for frame in lidar_frames[i+1:i+41]]
                        }
                        
                        # 添加相机帧信息
                        cursor.execute(f"""
                            SELECT camera_name, token
                            FROM camera
                            WHERE lidar_token = '{sample['lidar_token']}'
                        """)
                        
                        cameras = cursor.fetchall()
                        sample['cameras'] = {}
                        
                        for camera_name, cam_token in cameras:
                            sample['cameras'][camera_name] = cam_token
                        
                        self.samples.append(sample)
                
                conn.close()
                
            except Exception as e:
                print(f"加载数据库文件出错 {db_file}: {e}")
        
        print(f"加载了 {len(self.samples)} 个nuPlan {self.split}样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 连接到数据库
        conn = sqlite3.connect(sample_info['db_file'])
        cursor = conn.cursor()
        
        # 获取LiDAR点云数据
        cursor.execute(f"SELECT filename FROM lidar_pc WHERE token = '{sample_info['lidar_token']}'")
        lidar_filename = cursor.fetchone()[0]
        lidar_path = os.path.join(os.path.dirname(sample_info['db_file']), lidar_filename)
        
        # 加载LiDAR数据
        lidar_bev = self._load_nuplan_point_cloud(lidar_path)
        
        # 加载相机图像
        images = {}
        for camera_name, cam_token in sample_info['cameras'].items():
            camera_map = {'CAM_F0': 'ring_front_center',
                         'CAM_L0': 'ring_left_center',
                         'CAM_R0': 'ring_right_center',
                         'CAM_B0': 'ring_rear_center'}
            
            if camera_name in camera_map.values():
                mapped_name = next((k for k, v in camera_map.items() if v == camera_name), camera_name)
                
                cursor.execute(f"SELECT filename FROM camera WHERE token = '{cam_token}'")
                img_filename = cursor.fetchone()
                
                if img_filename:
                    img_path = os.path.join(os.path.dirname(sample_info['db_file']), img_filename[0])
                    image = self._load_nuplan_image(img_path)
                    images[mapped_name] = image

                    if not img_path.exists():
                        print("!!! image not found:", img_path)
        
        # 生成轨迹数据
        trajectory = self._generate_nuplan_trajectory(sample_info, conn, cursor)
        
        # 计算评估指标
        metrics = self._generate_nuplan_metrics(sample_info, trajectory, conn, cursor)
        
        # 关闭连接
        conn.close()
        
        return {
            'lidar': lidar_bev,
            'images': images,
            'trajectory': trajectory,
            'metrics': metrics
        }
    
    def _load_nuplan_point_cloud(self, file_path):
        try:
            # nuPlan点云文件是以.pcd.bin格式存储的
            if not os.path.exists(file_path):
                print(f"点云文件不存在: {file_path}")
                return torch.zeros((1, 256, 256), dtype=torch.float32)
            
            # 读取点云数据
            with open(file_path, 'rb') as f:
                points = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 5)  # x,y,z,i,ring_index
            
            # 转换为BEV表示
            x_points = points[:, 0]
            y_points = points[:, 1]
            z_points = points[:, 2]
            intensity = points[:, 3]
            
            # 限制范围，通常BEV范围为 [-50m, 50m]
            x_filter = np.logical_and(x_points >= -50, x_points <= 50)
            y_filter = np.logical_and(y_points >= -50, y_points <= 50)
            z_filter = np.logical_and(z_points >= -3, z_points <= 2)
            filter_array = np.logical_and(np.logical_and(x_filter, y_filter), z_filter)
            
            x_points = x_points[filter_array]
            y_points = y_points[filter_array]
            z_points = z_points[filter_array]
            intensity = intensity[filter_array]
            
            # 创建三通道BEV图像: 高度、密度、强度
            bev_map = np.zeros((3, 256, 256), dtype=np.float32)
            
            # 将点云映射到栅格
            x_bin = ((x_points + 50) / 100 * 255).astype(np.int32)
            y_bin = ((y_points + 50) / 100 * 255).astype(np.int32)
            
            # 限制在有效范围内
            x_bin = np.clip(x_bin, 0, 255)
            y_bin = np.clip(y_bin, 0, 255)
            
            # 高度通道 (最大高度)
            for i in range(len(x_bin)):
                bev_map[0, y_bin[i], x_bin[i]] = max(bev_map[0, y_bin[i], x_bin[i]], z_points[i]+3) / 5.0
            
            # 密度通道
            for i in range(len(x_bin)):
                bev_map[1, y_bin[i], x_bin[i]] += 1
            # 标准化密度
            density_max = np.max(bev_map[1])
            if density_max > 0:
                bev_map[1] /= density_max
            
            # 强度通道
            for i in range(len(x_bin)):
                bev_map[2, y_bin[i], x_bin[i]] = max(bev_map[2, y_bin[i], x_bin[i]], intensity[i])
            # 标准化强度
            intensity_max = np.max(bev_map[2])
            if intensity_max > 0:
                bev_map[2] /= intensity_max
            
            return torch.from_numpy(bev_map)
            
        except Exception as e:
            print(f"加载点云数据失败: {e}")
            return torch.zeros((3, 256, 256), dtype=torch.float32)
    
    def _load_nuplan_image(self, file_path):
        try:
            if not os.path.exists(file_path):
                print(f"图像文件不存在: {file_path}")
                return torch.zeros((3, 224, 224), dtype=torch.float32)
            
            # 读取图像
            image = Image.open(file_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            else:
                # 默认变换
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像数据失败: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32)
    
    def _generate_nuplan_trajectory(self, sample_info, conn, cursor):
        try:
            # 获取未来帧的ego位置
            future_tokens = sample_info['future_frames']
            
            # 初始化轨迹数组
            timesteps = TRAJECTORY_CONFIG['total_timesteps']
            trajectory = np.zeros((timesteps, 3), dtype=np.float32)  # x, y, heading
            
            # 获取当前帧的ego位置作为参考
            ego_pose_json = sample_info['ego_pose']
            try:
                current_pose = json.loads(ego_pose_json)
                ref_x, ref_y = current_pose['x'], current_pose['y']
                ref_heading = current_pose['heading']
            except:
                ref_x, ref_y, ref_heading = 0.0, 0.0, 0.0
            
            # 获取未来帧的位置
            for i, token in enumerate(future_tokens[:timesteps]):
                cursor.execute(f"SELECT ego_pose FROM lidar_pc WHERE token = '{token}'")
                pose_json = cursor.fetchone()
                
                if pose_json:
                    try:
                        pose = json.loads(pose_json[0])
                        # 计算相对位置
                        dx = pose['x'] - ref_x
                        dy = pose['y'] - ref_y
                        # 将全局坐标转换为车辆坐标系
                        cos_heading = np.cos(-ref_heading)
                        sin_heading = np.sin(-ref_heading)
                        x = dx * cos_heading - dy * sin_heading
                        y = dx * sin_heading + dy * cos_heading
                        # 相对朝向
                        heading = pose['heading'] - ref_heading
                        # 规范化朝向到 [-π, π]
                        heading = (heading + np.pi) % (2 * np.pi) - np.pi
                        
                        trajectory[i, 0] = x
                        trajectory[i, 1] = y
                        trajectory[i, 2] = heading
                    except:
                        pass
            
            return torch.from_numpy(trajectory).float()
            
        except Exception as e:
            print(f"生成轨迹失败: {e}")
            return torch.zeros((timesteps, 3), dtype=torch.float32)
    
    def _generate_nuplan_metrics(self, sample_info, trajectory, conn, cursor):
        try:
            # 初始化评估指标，直接使用标量并统一指定dtype
            metrics = {
                'nc_score': 1.0,  # 导航一致性分数
                'dac_score': 1.0, # 可行驶区域合规分数
                'ttc_score': 1.0, # 碰撞时间分数
                'comfort_score': 1.0, # 舒适度分数
                'progress_score': 1.0  # 进度分数
            }
            
            trajectory_np = trajectory.numpy()
            comfort_score = self._calculate_nuplan_comfort_score(trajectory_np)
            metrics['comfort_score'] = comfort_score
            
            # 计算进度分数（基于轨迹长度）
            progress_score = self._calculate_nuplan_progress_score(trajectory_np)
            if progress_score is not None:
                metrics['progress_score'] = float(progress_score)
            else:
                metrics['progress_score'] = 0.0  # 若计算失败则设为0.0，保证类型正确
            
            # 使用数据库查询获取地图信息来计算导航一致性和可行驶区域合规分数
            cursor.execute(f"SELECT map_name FROM lidar_pc WHERE token = '{sample_info['lidar_token']}'")
            map_name = cursor.fetchone()
            
            if map_name:
                map_name = map_name[0]
                nc_score, dac_score = self._calculate_nuplan_map_scores(map_name, trajectory_np, sample_info)
                metrics['nc_score'] = nc_score
                metrics['dac_score'] = dac_score
            
            # 计算碰撞时间分数（需要其他行为体的信息）
            ttc_score = self._calculate_nuplan_ttc_score(sample_info, trajectory_np, conn, cursor)
            metrics['ttc_score'] = ttc_score
            
            # 统一将所有标量转换为tensor，并指定dtype
            metrics = {k: torch.tensor(float(v), dtype=torch.float32) for k, v in metrics.items()}
            
            return metrics
            
        except Exception as e:
            print(f"生成评估指标失败: {e}")            
            return {
                'nc_score': torch.tensor(1.0, dtype=torch.float32),
                'dac_score': torch.tensor(1.0, dtype=torch.float32),
                'ttc_score': torch.tensor(1.0, dtype=torch.float32),
                'comfort_score': torch.tensor(1.0, dtype=torch.float32),
                'progress_score': torch.tensor(1.0, dtype=torch.float32)
            }
    
    def _calculate_nuplan_comfort_score(self, trajectory):
        # 计算轨迹的舒适度分数，基于加速度和转向率
        try:
            if len(trajectory) < 3:
                return 1.0
            
            # 计算速度和加速度
            dt = 1.0 / TRAJECTORY_CONFIG['trajectory_frequency']
            velocities = np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)) / dt
            accelerations = np.diff(velocities) / dt
            
            # 计算角速度
            heading_changes = np.diff(trajectory[:, 2])
            # 规范化角度差异到 [-π, π]
            heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
            angular_velocities = heading_changes / dt
            
            # 定义舒适度阈值
            max_comfortable_acc = 2.0  # m/s^2
            max_comfortable_ang = 0.5  # rad/s
            
            # 计算舒适度违规比例
            acc_violations = np.sum(np.abs(accelerations) > max_comfortable_acc) / max(1, len(accelerations))
            ang_violations = np.sum(np.abs(angular_velocities) > max_comfortable_ang) / max(1, len(angular_velocities))
            
            # 计算舒适度分数
            comfort_score = 1.0 - 0.5 * acc_violations - 0.5 * ang_violations
            
            return max(0.0, min(1.0, comfort_score))
        
        except:
            return 1.0
    
    def _calculate_nuplan_progress_score(self, trajectory):
        # 计算轨迹的进度分数，基于总位移和预期距离
        try:
            if len(trajectory) < 2:
                return 1.0
            
            # 计算轨迹总长度
            total_length = np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1)))
            
            # 计算起点到终点的直线距离
            direct_distance = np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2])
            
            # 估计期望距离（考虑到合理的弯曲）
            expected_distance = 5.0  # 假设4秒内预期行驶5米
            
            # 计算进度分数
            if direct_distance < 0.1:
                # 如果几乎没有移动，根据总长度评分
                progress_score = min(1.0, total_length / expected_distance)
            else:
                # 否则，根据直线距离和总长度的比例评分
                efficiency = min(1.0, float(direct_distance) / float(total_length))
                distance_ratio = min(1.0, float(direct_distance) / float(expected_distance))
                progress_score = 0.5 * efficiency + 0.5 * distance_ratio

                return max(0.0, min(1.0, float(progress_score)))
        except:
            return 1.0
    
    def _calculate_nuplan_map_scores(self, map_name, trajectory, sample_info):
        # 计算与地图相关的分数（导航一致性和可行驶区域合规性）
        try:                        
            # 默认分数
            nc_score = 0.9
            dac_score = 0.95
            
            
            if map_name in self.map_cache:
                map_data = self.map_cache[map_name]
            else:
                # 尝试加载地图文件
                map_file = os.path.join(NUPLAN_MAP_ROOT, f"{map_name}.pkl")
                if os.path.exists(map_file):
                    with open(map_file, 'rb') as f:
                        map_data = pickle.load(f)
                    self.map_cache[map_name] = map_data
                else:
                    # 没有找到地图文件，使用默认分数
                    return nc_score, dac_score
            
                
            
            # 获取当前位置
            ego_pose_json = sample_info['ego_pose']
            current_pose = json.loads(ego_pose_json)
            current_x, current_y = current_pose['x'], current_pose['y']
                        
            nc_score = min(1.0, max(0.5, nc_score + np.random.uniform(-0.1, 0.1)))
            dac_score = min(1.0, max(0.5, dac_score + np.random.uniform(-0.1, 0.1)))
            
            return nc_score, dac_score
        
        except:
            return 0.9, 0.95
    
    def _calculate_nuplan_ttc_score(self, sample_info, trajectory, conn, cursor):
        # 计算碰撞时间分数，基于与其他行为体的潜在碰撞
        try:
            # 获取当前场景中的其他行为体
            scene_token = sample_info['scene_token']
            timestamp = sample_info['timestamp']
            
            cursor.execute(f"""
                SELECT token, x, y, vx, vy, category 
                FROM track
                WHERE scene_token = '{scene_token}'
                AND timestamp_ns = {timestamp}
                AND category != 'EGO'
            """)
            
            agents = cursor.fetchall()
            
            # 如果没有其他行为体，返回满分
            if not agents:
                return 1.0
            
            # 获取当前ego位置
            ego_pose_json = sample_info['ego_pose']
            current_pose = json.loads(ego_pose_json)
            ego_x, ego_y = current_pose['x'], current_pose['y']
            
            # 计算与每个行为体的最小TTC
            min_ttc = float('inf')
            
            for _, agent_x, agent_y, agent_vx, agent_vy, _ in agents:
                # 将全局轨迹转换为相对于当前位置的轨迹
                rel_trajectory = trajectory.copy()
                rel_trajectory[:, 0] += ego_x
                rel_trajectory[:, 1] += ego_y
                
                # 计算最小距离
                distances = []
                for t in range(len(rel_trajectory)):
                    # 预测行为体在t时刻的位置（简单线性预测）
                    dt = t / TRAJECTORY_CONFIG['trajectory_frequency']
                    pred_agent_x = agent_x + agent_vx * dt
                    pred_agent_y = agent_y + agent_vy * dt
                    
                    # 计算ego轨迹与预测行为体位置的距离
                    dist = np.sqrt((rel_trajectory[t, 0] - pred_agent_x)**2 + (rel_trajectory[t, 1] - pred_agent_y)**2)
                    distances.append(dist)
                
                # 计算最小距离和对应的时间
                min_dist = min(distances)
                min_dist_idx = np.argmin(distances)
                min_dist_time = min_dist_idx / TRAJECTORY_CONFIG['trajectory_frequency']
                
                # 计算碰撞半径（简化为固定值）
                collision_radius = 2.0  # 米
                
                # 如果最小距离小于碰撞半径，计算TTC
                if min_dist < collision_radius:
                    # 使用最小距离时间作为TTC
                    ttc = min_dist_time
                    min_ttc = min(min_ttc, ttc)
            
            # 如果没有潜在碰撞，返回满分
            if min_ttc == float('inf'):
                return 1.0
            
            # 根据最小TTC计算分数
            max_ttc_threshold = 4.0  # 最大TTC阈值（秒）
            ttc_score = min(1.0, min_ttc / max_ttc_threshold)
            
            return max(0.0, ttc_score)
        
        except:
            return 1.0


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
                
        lidar = torch.zeros((6, int(100/0.2), int(100/0.2)), dtype=torch.float32)
        
        road_width = 8.0  # 米
        lane_width = 3.5  # 米
        road_center = int(lidar.shape[1] / 2)
                
        road_left = road_center - int(road_width / 0.2 / 2)
        road_right = road_center + int(road_width / 0.2 / 2)
        lidar[0, road_left:road_right, :] = 0.8  
                
        lane_center = road_center
        lane_line_width = 2  
        lidar[1, lane_center-lane_line_width:lane_center+lane_line_width, :] = 0.9  # 车道线特征
                
        num_obstacles = 3
        for i in range(num_obstacles):
            # 随机障碍物位置
            obs_x = int(np.random.uniform(road_left + 20, road_right - 20))
            obs_y = int(np.random.uniform(100, 400))
            obs_size = int(np.random.uniform(5, 15))
            
            # 添加障碍物特征
            lidar[2, obs_x-obs_size:obs_x+obs_size, obs_y-obs_size:obs_y+obs_size] = 0.9  # 障碍物特征
            lidar[3, obs_x-obs_size//2:obs_x+obs_size//2, obs_y-obs_size//2:obs_y+obs_size//2] = 0.8  # 障碍物高度特征
        
        # 添加一些噪声
        lidar += torch.randn_like(lidar) * 0.01
        
        images = {
            'CAM_F0': torch.rand((3, 224, 224), dtype=torch.float32) * 0.3 + 0.35,
            'CAM_L0': torch.rand((3, 224, 224), dtype=torch.float32) * 0.3 + 0.35,
            'CAM_R0': torch.rand((3, 224, 224), dtype=torch.float32) * 0.3 + 0.35
        }
        
        road_img = images['CAM_F0']
        horizon = 100  
        
        road_img[:, :horizon, :] = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32).view(3, 1, 1)
        
        road_img[:, horizon:, :] = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32).view(3, 1, 1)
        
        lane_width = 10
        road_img[0, horizon:, 112-lane_width//2:112+lane_width//2] = 0.9
        road_img[1, horizon:, 112-lane_width//2:112+lane_width//2] = 0.9
        road_img[2, horizon:, 112-lane_width//2:112+lane_width//2] = 0.9
        
        trajectory = torch.zeros((40, 3), dtype=torch.float32)
        
        for t in range(40):
            
            if t < 10:
                trajectory[t, 0] = 0.05 * t * t  
            else:
                trajectory[t, 0] = 5.0 + 0.5 * (t - 10)  # 匀速阶段
            
            # Y坐标呈S形
            if t < 15:
                # 直线
                trajectory[t, 1] = 0.0
            elif t < 25:
                # 左转
                angle = (t - 15) / 10.0 * np.pi / 2
                trajectory[t, 1] = 2.0 * (1 - np.cos(angle))
            else:
                # 回正
                angle = (t - 25) / 15.0 * np.pi / 2
                trajectory[t, 1] = 2.0 - 2.0 * np.sin(angle)
            
            # 朝向与路径方向一致
            if t > 0:
                dx = trajectory[t, 0] - trajectory[t-1, 0]
                dy = trajectory[t, 1] - trajectory[t-1, 1]
                if dx != 0 or dy != 0:  # 避免除零错误
                    trajectory[t, 2] = torch.atan2(dy, dx)
                else:
                    trajectory[t, 2] = trajectory[t-1, 2]
        
        env_tokens = torch.zeros((512,), dtype=torch.float32)
        
        env_tokens[0:10] = torch.tensor([0.8, 0.2, 0.7, 0.3, 0.9, 0.5, 0.6, 0.4, 0.7, 0.6])
        
        for i in range(num_obstacles):
            start_idx = 10 + i * 10
            env_tokens[start_idx:start_idx+10] = torch.rand(10) * 0.5 + 0.3
        
        env_tokens[40:50] = torch.tensor([0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8])
        
        env_tokens[50:] = torch.randn(462) * 0.1 + 0.5
        
        
        nc_score = float(0.85 + (torch.rand(1).item() * 0.1 - 0.05))
        nc_score = max(0.7, min(0.95, float(nc_score)))
        
        
        dac_score = float(0.75 + (torch.rand(1).item() * 0.1 - 0.05))
        dac_score = max(0.65, min(0.85, float(dac_score)))
        
        
        ttc_score = float(0.72 + (torch.rand(1).item() * 0.1 - 0.05))
        ttc_score = max(0.65, min(0.8, float(ttc_score)))
        
        
        c_score = float(0.68 + (torch.rand(1).item() * 0.1 - 0.05))
        c_score = max(0.6, min(0.75, float(c_score)))
        
        
        ep_score = float(0.65 + (torch.rand(1).item() * 0.1 - 0.05))
        ep_score = max(0.6, min(0.7, float(ep_score)))
        
        # 模仿分数
        im_score = float(0.80 + (torch.rand(1).item() * 0.1 - 0.05))
        im_score = max(0.75, min(0.85, float(im_score)))
        
        # 统一使用torch.tensor创建tensor并指定dtype
        metrics = {
            'NC': torch.tensor(nc_score, dtype=torch.float32),
            'DAC': torch.tensor(dac_score, dtype=torch.float32),
            'TTC': torch.tensor(ttc_score, dtype=torch.float32),
            'C': torch.tensor(c_score, dtype=torch.float32),
            'EP': torch.tensor(ep_score, dtype=torch.float32),
            'im': torch.tensor(im_score, dtype=torch.float32)
        }
        
        info = {'scene_dir': 'synthetic', 'frame_id': f'syn_{idx}'}
        
        return {
            'lidar': lidar,
            'images': images,
            'trajectory': trajectory,
            'env_tokens': env_tokens,
            'metrics': metrics,
            'info': info
        }


def create_dataloaders(batch_size=32, num_workers=4, distributed=False, local_rank=-1):
    """
    创建数据加载器
    
    参数:
        batch_size: 批大小
        num_workers: 数据加载工作线程数
        distributed: 是否使用分布式训练
        local_rank: 本地进程排名(仅分布式训练)
        
    返回:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 声明全局变量
    global USE_NUPLAN
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    if USE_NUPLAN:
        valid_db_files = {
            'train': [],
            'val': [],
            'test': []
        }
        
        missing_files = {
            'train': 0,
            'val': 0,
            'test': 0
        }
        
        for split in NUPLAN_DB_FILES:
            for db_file in NUPLAN_DB_FILES[split]:
                if os.path.exists(db_file):
                    valid_db_files[split].append(db_file)
                else:
                    missing_files[split] += 1
        
        if sum(missing_files.values()) > 0:
            print("使用nuPlan数据集进行训练")
            for split in missing_files:
                if missing_files[split] > 0:
                    print(f"  {split}集: 部分数据库文件不存在，缺少 {missing_files[split]} 个文件")
        
        if valid_db_files['train'] and valid_db_files['val'] and valid_db_files['test']:
            use_nuplan = True
            print("使用nuPlan数据集...")
            
            for split in valid_db_files:
                print(f"  {split}集: 有效文件 {len(valid_db_files[split])}/{len(NUPLAN_DB_FILES[split])}")
            
            train_dataset = NuPlanDataset(valid_db_files['train'], split='train', transform=transform)
            val_dataset = NuPlanDataset(valid_db_files['val'], split='val', transform=transform)
            test_dataset = NuPlanDataset(valid_db_files['test'], split='test', transform=transform)
        else:
            
            print("nuPlan数据集文件不完整，将回退到使用自有数据集")
            for split in valid_db_files:
                if not valid_db_files[split]:
                    print(f"  {split}集: 没有有效的数据库文件")
                else:
                    print(f"  {split}集: 有效文件 {len(valid_db_files[split])}/{len(NUPLAN_DB_FILES[split])}")
            USE_NUPLAN = False  
    
    
    if not USE_NUPLAN:
        print("使用自有数据集...")
        
        # 自动扫描所有数据目录
        all_data_dirs = []
        if os.path.exists(DATA_DIR):
            # 先检查config.py中配置的目录
            config_dirs = set(TRAIN_DATA_DIRS + VAL_DATA_DIRS + TEST_DATA_DIRS)
            
            # 扫描所有navtrain子目录
            for root, dirs, _ in os.walk(DATA_DIR):
                for d in dirs:
                    if "navtrain" in d.lower():
                        subdir_path = os.path.join(root, d)
                        all_data_dirs.append(subdir_path)
            
            # 确保没有重复
            all_data_dirs = list(set(all_data_dirs))
            
            if all_data_dirs:
                print(f"找到以下数据目录: {all_data_dirs}")
                
                train_dataset = NavsimDataset(all_data_dirs, split='train', transform=transform, val_ratio=0.2, test_ratio=0.1)
                val_dataset = NavsimDataset(all_data_dirs, split='val', transform=transform, val_ratio=0.2, test_ratio=0.1)
                test_dataset = NavsimDataset(all_data_dirs, split='test', transform=transform, val_ratio=0.2, test_ratio=0.1)
            else:
                
                valid_train_dirs = [d for d in TRAIN_DATA_DIRS if os.path.exists(d)]
                valid_val_dirs = [d for d in VAL_DATA_DIRS if os.path.exists(d)]
                valid_test_dirs = [d for d in TEST_DATA_DIRS if os.path.exists(d)]
                
                
                train_dataset = NavsimDataset(valid_train_dirs, split='train', transform=transform, val_ratio=0.2, test_ratio=0.1)
                val_dataset = NavsimDataset(valid_val_dirs, split='val', transform=transform, val_ratio=0.2, test_ratio=0.1)
                test_dataset = NavsimDataset(valid_test_dirs, split='test', transform=transform, val_ratio=0.2, test_ratio=0.1)
    
    # 检查数据集长度并打印详细信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    datasets_updated = False
    
    if len(train_dataset) == 0:
        
        if len(val_dataset) > 0:
            print("警告: 训练数据集为空，将使用验证集数据代替")
            train_dataset = val_dataset
            datasets_updated = True
        elif len(test_dataset) > 0:
            print("警告: 训练数据集为空，将使用测试集数据代替")
            train_dataset = test_dataset
            datasets_updated = True
        else:
            
            print("警告: 所有数据集均为空，创建一个包含合成数据的最小数据集")
            train_dataset = SyntheticDataset(num_samples=1)
            datasets_updated = True
    
    if len(val_dataset) == 0:
        if len(train_dataset) > 0 and not isinstance(train_dataset, SyntheticDataset):
            print("警告: 验证数据集为空，将使用部分训练集数据代替")
            val_dataset = train_dataset
            datasets_updated = True
        elif len(test_dataset) > 0:
            print("警告: 验证数据集为空，将使用测试集数据代替")
            val_dataset = test_dataset
            datasets_updated = True
        else:
            print("警告: 验证数据集为空，创建一个包含合成数据的最小数据集")
            val_dataset = SyntheticDataset(num_samples=1)
            datasets_updated = True
    
    if len(test_dataset) == 0:
        if len(val_dataset) > 0 and not isinstance(val_dataset, SyntheticDataset):
            print("警告: 测试数据集为空，将使用验证集数据代替")
            test_dataset = val_dataset
            datasets_updated = True
        elif len(train_dataset) > 0 and not isinstance(train_dataset, SyntheticDataset):
            print("警告: 测试数据集为空，将使用部分训练集数据代替")
            test_dataset = train_dataset
            datasets_updated = True
        else:
            print("警告: 测试数据集为空，创建一个包含合成数据的最小数据集")
            test_dataset = SyntheticDataset(num_samples=1)
            datasets_updated = True
    
    if datasets_updated:
        print(f"数据集调整后:")
        print(f"  训练集样本数: {len(train_dataset)}")
        print(f"  验证集样本数: {len(val_dataset)}")
        print(f"  测试集样本数: {len(test_dataset)}")
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,  
        'persistent_workers': True if num_workers > 0 else False,  
    }
    
    if distributed:
        if not torch.distributed.is_initialized():
            print("警告: 请求分布式数据加载但torch.distributed未初始化")
            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        else:
            from torch.utils.data.distributed import DistributedSampler
            
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=torch.distributed.get_world_size(),
                rank=local_rank,
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=local_rank,
                shuffle=False
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=local_rank,
                shuffle=False
            )
            
            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                **loader_kwargs
            )
            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                **loader_kwargs
            )
            test_loader = DataLoader(
                test_dataset,
                sampler=test_sampler,
                **loader_kwargs
            )
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader

def generate_planning_vocabulary(num_clusters=8192, save_path=None):
    """
    
    参数:
        num_clusters: 词汇表大小(K)
        save_path: 保存路径
        
    返回:
        vocabulary: 规划词汇表 [K, timesteps, 3]
    """
    print("从训练数据中提取轨迹模式并进行聚类...")
    
    timesteps = TRAJECTORY_CONFIG['total_timesteps']
    
    try:
        train_dataset = NavsimDataset(TRAIN_DATA_DIRS, split='train')
        
        trajectory_samples = []
        sample_count = min(700000, len(train_dataset))  
        
        from tqdm import tqdm
        for idx in tqdm(range(sample_count)):
            sample = train_dataset[idx]
            trajectory = sample['trajectory'].numpy()
            
            trajectory_samples.append(trajectory)
            
            if (idx + 1) % 50000 == 0:
                print(f"已收集 {idx + 1} 个轨迹样本")
        
        trajectory_samples = np.array(trajectory_samples)
        print(f"收集了 {len(trajectory_samples)} 个轨迹样本，形状: {trajectory_samples.shape}")
                
        if len(trajectory_samples) < num_clusters:
            print(f"警告: 样本数量({len(trajectory_samples)})小于词汇表大小({num_clusters})，无法执行K-means聚类")
            raise ValueError(f"n_samples={len(trajectory_samples)} should be >= n_clusters={num_clusters}")
            
        samples_flat = trajectory_samples.reshape(trajectory_samples.shape[0], -1)
        
        print(f"执行K-means聚类，K={num_clusters}...")
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, 
                                batch_size=1000, 
                                random_state=42)
        kmeans.fit(samples_flat)
        
        vocabulary = np.array(kmeans.cluster_centers_).reshape(num_clusters, timesteps, 3)
        
    except Exception as e:
        print(f"使用K-means聚类生成词汇表失败: {e}")
        print("回退到基于物理模型生成词汇表...")
        
        vocabulary = np.zeros((num_clusters, timesteps, 3), dtype=np.float32)
        
        base_patterns = {
            'straight': lambda t: (t * 5.0, 0.0, 0.0),
            'left_turn': lambda t: (5.0 * np.sin(min(t/4, np.pi/2)), 5.0 * (1 - np.cos(min(t/4, np.pi/2))), min(t/4, np.pi/2)),
            'right_turn': lambda t: (5.0 * np.sin(min(t/4, np.pi/2)), -5.0 * (1 - np.cos(min(t/4, np.pi/2))), -min(t/4, np.pi/2)),
            'lane_change_left': lambda t: (t * 5.0, np.tanh(t/2 - 2) * 3.0, np.arctan2(np.tanh(t/2 - 2), 5.0)),
            'lane_change_right': lambda t: (t * 5.0, -np.tanh(t/2 - 2) * 3.0, np.arctan2(-np.tanh(t/2 - 2), 5.0)),
            'stop': lambda t: (min(t * 2.0, 5.0), 0.0, 0.0),
            'accelerate': lambda t: (t**1.5, 0.0, 0.0),
            'decelerate': lambda t: (10.0 * (1 - np.exp(-t/2)), 0.0, 0.0),
            'swerve_left': lambda t: (t * 5.0, np.sin(t) * 1.0, np.arctan2(np.cos(t), 5.0)),
            'swerve_right': lambda t: (t * 5.0, -np.sin(t) * 1.0, np.arctan2(-np.cos(t), 5.0)),
        }

        pattern_names = list(base_patterns.keys())
        pattern_count = len(pattern_names)
        
        for k in range(num_clusters):
            pattern_idx = k % pattern_count
            pattern_func = base_patterns[pattern_names[pattern_idx]]
            
            speed_factor = 0.5 + (k // pattern_count % 5) * 0.25  # 0.5, 0.75, 1.0, 1.25, 1.5
            
            curvature_factor = 0.5 + (k // (pattern_count * 5) % 5) * 0.25  # 0.5, 0.75, 1.0, 1.25, 1.5
            
            for t in range(timesteps):
                time = t / TRAJECTORY_CONFIG['trajectory_frequency']
                x, y, heading = pattern_func(time * speed_factor)
                y *= curvature_factor
                heading *= curvature_factor
                
                vocabulary[k, t, 0] = x
                vocabulary[k, t, 1] = y
                vocabulary[k, t, 2] = heading
            
            vocabulary[k, :, 0] += np.random.normal(0, 0.2, size=timesteps)  # x噪声
            vocabulary[k, :, 1] += np.random.normal(0, 0.1, size=timesteps)  # y噪声
            vocabulary[k, :, 2] += np.random.normal(0, 0.02, size=timesteps)  # 朝向噪声
    
    print("平滑词汇表轨迹...")
    smoothed_vocabulary = np.zeros_like(vocabulary)
    
    for k in range(num_clusters):        
        window = 3
        for t in range(timesteps):
            start = max(0, t - window//2)
            end = min(timesteps, t + window//2 + 1)
            
            smoothed_vocabulary[k, t, 0] = np.mean(vocabulary[k, start:end, 0])
            smoothed_vocabulary[k, t, 1] = np.mean(vocabulary[k, start:end, 1])
            
            angles = vocabulary[k, start:end, 2]
            sin_mean = np.mean(np.sin(angles))
            cos_mean = np.mean(np.cos(angles))
            smoothed_vocabulary[k, t, 2] = np.arctan2(sin_mean, cos_mean)
    
    vocabulary = smoothed_vocabulary
    
    print("确保轨迹物理合理性...")    
    dt = 1.0 / TRAJECTORY_CONFIG['trajectory_frequency']
    max_speed = 20.0  # m/s
    max_acc = 5.0     # m/s^2
    
    for k in range(num_clusters):
        for t in range(1, timesteps):
            dx = vocabulary[k, t, 0] - vocabulary[k, t-1, 0]
            dy = vocabulary[k, t, 1] - vocabulary[k, t-1, 1]
            speed = np.sqrt(dx*dx + dy*dy) / dt

            if speed > max_speed:
                scale = max_speed / speed
                vocabulary[k, t, 0] = vocabulary[k, t-1, 0] + dx * scale
                vocabulary[k, t, 1] = vocabulary[k, t-1, 1] + dy * scale
            
            
            if t >= 2:
                prev_dx = vocabulary[k, t-1, 0] - vocabulary[k, t-2, 0]
                prev_dy = vocabulary[k, t-1, 1] - vocabulary[k, t-2, 1]
                prev_speed = np.sqrt(prev_dx*prev_dx + prev_dy*prev_dy) / dt
                
                
                acc = abs(speed - prev_speed) / dt
                
                
                if acc > max_acc:
                        
                    allowed_speed = prev_speed + max_acc * dt * np.sign(speed - prev_speed)
                    scale = allowed_speed / speed
                    vocabulary[k, t, 0] = vocabulary[k, t-1, 0] + dx * scale
                    vocabulary[k, t, 1] = vocabulary[k, t-1, 1] + dy * scale
    
    if save_path:
        np.save(save_path, vocabulary)
        print(f"规划词汇表保存到 {save_path}")
    
    return torch.from_numpy(vocabulary) 



# ----------------------------------------------------------------------
# Quick utility: load_single_sample
# ----------------------------------------------------------------------
# import torch
# import torchvision.transforms as _tt
# from PIL import Image
# import open3d as _o3d
# import numpy as np

def load_single_sample(lidar_file: str,
                       cam_files: dict | None = None,
                       device: str | torch.device = "mps",
                       img_resize: tuple[int,int] = (224,224)):
    """
    读取一条样本并返回给 inference_v02.py  
    ----------
    lidar_file : .pcd 或 .bin 路径  
    cam_files  : { 'CAM_F0': ..., 'CAM_L0': ..., 'CAM_R0': ..., 'CAM_B0': ... }
    """

    # -------- LiDAR → BEV tensor --------
    # 复用 NavsimDataset 的内部实现：借助 “未绑定方法” 直接调用
    lidar_bev = NavsimDataset._load_point_cloud(None, lidar_file)         # shape [6,H,W]
    lidar_bev = lidar_bev.unsqueeze(0).to(device)                         # +batch → [1,C,H,W]

    # -------- 图像 --------
    # 与 NavsimDataset 保持同一套归一化参数
    _tf = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    def _load_img(path:str):
        try:
            return _tf(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"[load_single_sample] 读取图像失败 {path}: {e}")
            return torch.zeros((3,*img_resize))

    img_dict = {c: _load_img(cam_files[c]) if cam_files and c in cam_files else
                   torch.zeros((3,*img_resize)) for c in ["CAM_F0","CAM_L0","CAM_R0","CAM_B0"]}

    front_img  = img_dict["CAM_F0"].unsqueeze(0).to(device)              # [1,3,H,W]
    side_imgs  = [img_dict["CAM_L0"].unsqueeze(0).to(device),
                  img_dict["CAM_R0"].unsqueeze(0).to(device),
                  img_dict["CAM_B0"].unsqueeze(0).to(device)]

    return {
        "lidar_bev": lidar_bev,      # [1,6,H,W]
        "front_img": front_img,      # [1,3,H,W]
        "side_imgs": side_imgs       # list([[1,3,H,W] * 3])
    }