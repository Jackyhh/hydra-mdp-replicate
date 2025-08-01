# inference_v02.py
import os, torch, argparse
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from hydra_mdp import build_hydra_mdp
from data_utils import load_single_sample          # 你自己的读取 util
from config import CHECKPOINT_DIR, setup_device

# ---------- 1. 运行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=os.path.join(CHECKPOINT_DIR,'latest_checkpoint.pth'))
parser.add_argument('--sample', 
                            default='数据/navtrain_current_4/2021.06.09.14.03.17_veh-12_02112_02202/MergedPointCloud/e9996ea8bb7b5f4e.pcd',
                            # required=True, 
                            help='路径或 ID，用于测试的那帧/批数据')
args = parser.parse_args()

# ---------- 2. 设备 & 模型 ----------
device = setup_device()
model  = build_hydra_mdp().to(device)
ckpt   = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)   # 已在 HydraMDP 内处理 vocabulary 键
model.eval()

# ---------- 3. 读入一帧或一批数据 ----------
with torch.no_grad():
    # batch = load_single_sample(args.sample, device)   # 返回 dict，字段同 trainer 里

    batch = load_single_sample(args.sample)           # ✔ 只传路径

    # batch = load_single_sample(args.sample)
    print("DEBUG – keys in batch:", batch.keys())


    # 如果函数内部没自动 .to(device)，这里统一搬一下
    def _to_dev(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    batch = {k: _to_dev(v) if not isinstance(v, dict)
             else {kk: _to_dev(vv) for kk, vv in v.items()}
             for k, v in batch.items()}


    # lidar_bev  = batch['lidar']                       # [B,6,H,W]
    # front_img  = batch['images']['CAM_F0']            # [B,3,224,224]
    # side_imgs  = [batch['images'].get(cam) for cam in ['CAM_L0','CAM_R0','CAM_B0']]

    # load_single_sample 已经返回好了这三个 key
    lidar_bev = batch['lidar_bev']     # [B,6,H,W]
    front_img = batch['front_img']     # [B,3,224,224]
    side_imgs = batch['side_imgs']     # list，长度 3（或带 None 的占位）


    outputs    = model(lidar_bev, front_img, side_imgs, mode='test')

# ---------- 4. 可视化 ----------
trajectories = outputs['best_trajectories'].cpu()     # [B,40,3]  (x,y,heading)
for b in range(trajectories.size(0)):
    traj = trajectories[b].numpy()
    plt.figure(figsize=(6,6))
    plt.plot(traj[:,0],  traj[:,1],  marker='o', linewidth=2)   # xy 折线
    plt.quiver(traj[-1,0], traj[-1,1],                       # 在终点画 heading
               0.5*torch.cos(torch.tensor(traj[-1,2])),
               0.5*torch.sin(torch.tensor(traj[-1,2])),
               angles='xy', scale_units='xy', scale=1, color='r')
    plt.title(f'Sample {args.sample}  |  PDM={outputs["pdm_score"][b]:.3f}')
    plt.axis('equal');  plt.grid(True)
    plt.show()