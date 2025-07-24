import os
import argparse
import torch
import numpy as np
from config import setup_device, set_seed, CHECKPOINT_DIR, PLOTS_DIR, ROOT_DIR, USE_NUPLAN, NUPLAN_DB_FILES
from data_utils import create_dataloaders, generate_planning_vocabulary
from perception_network import build_perception_network
from trajectory_decoder import build_trajectory_decoder
from hydra_mdp import build_hydra_mdp, build_hydra_mdp_variants
from trainer import train_model, evaluate_model

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Hydra-MDP: 多模态规划与多目标Hydra蒸馏')
    
    # 主要命令
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='运行模式: train-训练, eval-评估')
    
    # 训练参数
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    
    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, help='分布式训练的本地排名')
    parser.add_argument('--world_size', type=int, default=None, help='分布式训练的总进程数')
    parser.add_argument('--dist_url', type=str, default='env://', help='分布式训练的URL')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='分布式训练的后端')
    
    # 数据集参数
    parser.add_argument('--use_nuplan', action='store_true', help='使用nuPlan数据集')
    parser.add_argument('--nuplan_data_root', type=str, default=None, help='nuPlan数据根目录')
    
    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=8192, help='规划词汇表大小')
    parser.add_argument('--model_type', type=str, default='single', choices=['single', 'ensemble'],
                        help='模型类型: single-单模型, ensemble-集成模型')
    

    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID')
    
    args = parser.parse_args()
    
    # 从环境变量读取额外配置
    if 'TRAINING_NUM_WORKERS' in os.environ:
        from config import TRAINING_CONFIG
        TRAINING_CONFIG['num_workers'] = int(os.environ['TRAINING_NUM_WORKERS'])
        print(f"从环境变量设置数据加载工作线程数: {TRAINING_CONFIG['num_workers']}")
    
    if 'DISABLE_MIXED_PRECISION' in os.environ:
        from config import TRAINING_CONFIG
        TRAINING_CONFIG['mixed_precision'] = False
        print("从环境变量禁用混合精度训练")
    
    if 'GRADIENT_ACCUMULATION_STEPS' in os.environ:
        from config import TRAINING_CONFIG
        TRAINING_CONFIG['gradient_accumulation_steps'] = int(os.environ['GRADIENT_ACCUMULATION_STEPS'])
        print(f"从环境变量设置梯度累积步数: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    
    return args

def prepare_environment(args):
    """准备运行环境"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 初始化分布式环境
    distributed = False
    
    if args.distributed or args.local_rank != -1:
        # 已通过torch.distributed.launch或torchrun启动
        if torch.cuda.is_available():
            if args.local_rank != -1:
                # 从环境变量获取排名
                torch.cuda.set_device(args.local_rank)
                device = torch.device("cuda", args.local_rank)
                
                # 初始化进程组
                if not torch.distributed.is_initialized():
                    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                        rank = int(os.environ['RANK'])
                        world_size = int(os.environ['WORLD_SIZE'])
                    else:
                        rank = args.local_rank
                        world_size = args.world_size or torch.cuda.device_count()
                    
                    torch.distributed.init_process_group(
                        backend=args.dist_backend,
                        init_method=args.dist_url,
                        world_size=world_size,
                        rank=rank
                    )
                    distributed = True
                    
                    if args.local_rank == 0:
                        print(f"初始化分布式环境: rank={rank}, world_size={world_size}")
            else:
                device = setup_device()
        else:
            device = setup_device()
    else:
        device = setup_device()
    
    if not distributed or args.local_rank == 0:
        print(f"使用设备: {device}")
    
    # 确保检查点目录存在
    if not distributed or args.local_rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 配置nuPlan数据集
    if args.use_nuplan:
        global USE_NUPLAN
        USE_NUPLAN = True
        
        if args.nuplan_data_root:
            global NUPLAN_DATA_ROOT
            NUPLAN_DATA_ROOT = args.nuplan_data_root
            
            # 更新数据库文件路径
            global NUPLAN_DB_FILES
            for split in NUPLAN_DB_FILES:
                for i, db_file in enumerate(NUPLAN_DB_FILES[split]):
                    NUPLAN_DB_FILES[split][i] = os.path.join(NUPLAN_DATA_ROOT, os.path.basename(db_file))
    
    return device

def run_training(args):
    """运行模型训练"""
    print("开始模型训练...")
    
    # 如果指定了训练参数，覆盖默认值
    from config import TRAINING_CONFIG
    
    if args.batch_size is not None:
        TRAINING_CONFIG['batch_size'] = args.batch_size
    
    if args.epochs is not None:
        TRAINING_CONFIG['num_epochs'] = args.epochs
    
    if args.lr is not None:
        TRAINING_CONFIG['learning_rate'] = args.lr
    
    # 设置训练的GPU数量，包括MPS
    from config import has_gpu
    
    if has_gpu():
        if torch.cuda.is_available():
            TRAINING_CONFIG['num_gpus'] = min(TRAINING_CONFIG['num_gpus'], torch.cuda.device_count())
            print(f"训练将使用 {TRAINING_CONFIG['num_gpus']} 个CUDA GPU")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            TRAINING_CONFIG['num_gpus'] = 1  # MPS只有一个GPU
            print(f"训练将使用Apple MPS加速")
    else:
        TRAINING_CONFIG['num_gpus'] = 0
        print("未检测到可用GPU，将使用CPU训练")
    
    # 打印数据集信息
    if USE_NUPLAN:
        print("使用nuPlan数据集进行训练")
        for split, db_files in NUPLAN_DB_FILES.items():
            file_exists = [os.path.exists(db_file) for db_file in db_files]
            if all(file_exists):
                print(f"  {split}集: {len(db_files)}个数据库文件")
            else:
                missing_files = [db_file for db_file, exists in zip(db_files, file_exists) if not exists]
                print(f"  {split}集: 部分数据库文件不存在，缺少 {len(missing_files)} 个文件")
    else:
        print("使用自有数据集进行训练")
    
    # 运行训练
    from trainer import train_model
    train_model()

def run_evaluation(args):
    """运行模型评估"""
    print("开始模型评估...")
    
    # 检查点路径
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    
    # 确保检查点文件存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点 {checkpoint_path} 不存在")
        return
    
    # 打印数据集信息
    if USE_NUPLAN:
        print("使用nuPlan数据集进行评估")
    else:
        print("使用自有数据集进行评估")
    
    # 评估模型
    pdm_score, sub_scores = evaluate_model(checkpoint_path, model_type=args.model_type)
    
    # 保存结果
    results_file = os.path.join(CHECKPOINT_DIR, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"模型评估结果:\n")
        f.write(f"检查点: {checkpoint_path}\n")
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"PDM分数: {pdm_score:.6f}\n")
        f.write(f"子分数:\n")
        for k, v in sub_scores.items():
            f.write(f"  {k}: {v:.6f}\n")
    
    print(f"评估结果已保存到 {results_file}")
    

def generate_vocabulary(args):
    """生成规划词汇表"""
    print(f"生成规划词汇表，大小: {args.vocab_size}...")
    
    # 保存路径
    save_path = os.path.join(CHECKPOINT_DIR, 'planning_vocabulary.npy')
    
    # 生成词汇表
    vocabulary = generate_planning_vocabulary(
        num_clusters=args.vocab_size,
        save_path=save_path
    )
    
    print(f"词汇表已保存到 {save_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 准备环境
    device = prepare_environment(args)
    
    # 根据模式运行相应功能
    if args.mode == 'train':
        # 如果词汇表不存在，先生成词汇表
        vocab_path = os.path.join(CHECKPOINT_DIR, 'planning_vocabulary.npy')
        if not os.path.exists(vocab_path):
            generate_vocabulary(args)
        
        # 运行训练
        run_training(args)
    elif args.mode == 'eval':
        run_evaluation(args)
    else:
        print(f"错误: 未知的运行模式 {args.mode}")

if __name__ == "__main__":
    main() 