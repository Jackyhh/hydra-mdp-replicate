# inference_v02.py
import os, torch, argparse
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from hydra_mdp import build_hydra_mdp
from data_utils import load_single_sample          # 你自己的读取 util
from config import CHECKPOINT_DIR, setup_device
import numpy as np
import datetime


# ---------- 1. 运行参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=os.path.join(CHECKPOINT_DIR,'latest_checkpoint.pth'))
parser.add_argument('--sample', 
                            default='数据/navtrain_current_4/2021.06.09.14.03.17_veh-12_02112_02202/MergedPointCloud/027a1824acf056f0.pcd',
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
    # ---------- 4. 可视化（相机 + BEV 双视图） ----------
    from torchvision.transforms.functional import normalize

    # 取 batch=0 做演示
    traj   = outputs['best_trajectories'][0].cpu().numpy()   # [40,3]
    bev_mt = lidar_bev[0]                                    # [6,H,W]
    cam0   = front_img[0]                                    # [3,224,224]

    # ------ (1) 处理相机图像 ------
    # 把归一化的 Tensor 转回 [0,1] 方便 imshow
    cam0_vis = cam0.clone().cpu()
    # 先反标准化（mean/std 同 config）
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    cam0_vis = cam0_vis * std + mean
    cam0_vis = cam0_vis.permute(1,2,0).clamp(0,1).numpy()   # [H,W,3]

    # ------ (2) 处理 BEV ------
    # 这里用第 5 通道(下标 5) 的密度 / 强度都可以
    bev_img = bev_mt[5].cpu().numpy()        # [H,W]  取值 0~1
    H, W    = bev_img.shape
    x_range = (-50,50)   # 同 data_utils._load_point_cloud
    y_range = (-50,50)
    grid    = 0.2

    # 轨迹米坐标 → 像素坐标
    xs_pix = (traj[:,0] - x_range[0]) / grid
    ys_pix = (traj[:,1] - y_range[0]) / grid
    # 由于 matplotlib 以左上为 (0,0)，而我们想让 +y 朝上，imshow(origin='lower')
    # 所以直接用即可

    # ------ (3) 画图 ------
    fig, ax = plt.subplots(1,2, figsize=(12,6))

    # 左：相机 + 简易轨迹 (只画折线示意)
    ax[0].imshow(cam0_vis)
    ax[0].set_title('Front Camera (approx overlay)')
    # 简单把 3D 轨迹 x→横向像素，y→纵向像素粗略映射到画面中央
    # ——没有标定只能做示意，真正精确需要外参/内参
    h,w,_ = cam0_vis.shape
    x_cam = np.linspace(w*0.45, w*0.55, len(traj))  # 横向放在中间
    y_cam = np.linspace(h*0.7,  h*0.2,  len(traj))  # 从下往上
    ax[0].plot(x_cam, y_cam, 'r-', linewidth=2)
    ax[0].scatter(x_cam[-1], y_cam[-1], c='r')      # 终点
    ax[0].axis('off')

    # 右：BEV + 精确轨迹
    ax[1].imshow(bev_img, cmap='gray', origin='lower', extent=[x_range[0],x_range[1],y_range[0],y_range[1]])
    ax[1].plot(traj[:,0], traj[:,1], 'r-', linewidth=2)
    ax[1].scatter(traj[-1,0], traj[-1,1], c='r')
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('X (m)')
    ax[1].set_ylabel('Y (m)')
    ax[1].set_title(f'BEV with Trajectory  |  PDM={outputs["pdm_score"][0]:.3f}')

    plt.tight_layout()


    # --------- (A) 先保存，再 show ----------
    os.makedirs("vis_out", exist_ok=True)       # NEW
    basename   = os.path.splitext(os.path.basename(args.sample))[0]
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path  = f"vis_out/{basename}_{ts}.png" # NEW
    plt.savefig(save_path, dpi=200)             # NEW *****
    print(f"可视化结果已保存: {save_path}")

    ax[1].imshow(bev_img, cmap='gray', origin='lower')
    ax[1].plot(xs_pix, ys_pix, 'r-', linewidth=2)
    ax[1].scatter(xs_pix[-1], ys_pix[-1], c='r')


    ax[1].set_xlabel('X (pixels)')
    ax[1].set_ylabel('Y (pixels)')

    plt.show()