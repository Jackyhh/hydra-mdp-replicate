# inference_v02.py  – 批量点云可视化（仅 BEV 侧）––––––––––––––––––––––
import os, glob, argparse, datetime, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')                # ← 关掉交互后端，纯保存
import matplotlib.pyplot as plt
import torch
# import imageio.v2 as iio       # imageio 用来写 mp4
import imageio_ffmpeg as iio

from hydra_mdp import build_hydra_mdp
from data_utils import load_single_sample
from config     import CHECKPOINT_DIR, setup_device

# ------------------------------------------------ argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default=os.path.join(CHECKPOINT_DIR,'latest_checkpoint.pth'))
parser.add_argument('--sample',   help='单帧 PCD 文件（优先级高）')
parser.add_argument('--scene_dir',help='一个 scene 目录；脚本会批量处理其中的 MergedPointCloud/*.pcd')
parser.add_argument('--fps', type=int, default=8, help='视频帧率')
args = parser.parse_args()

if not args.sample and not args.scene_dir:
    parser.error('必须至少提供 --sample 或 --scene_dir 之一')

# ------------------------------------------------ device & model
device   = setup_device()
model    = build_hydra_mdp().to(device).eval()
ckpt     = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)

# ------------------------------------------------ 准备输出目录
os.makedirs('vis_out/frames', exist_ok=True)
today  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
video_path = f"vis_out/{os.path.basename(args.scene_dir or 'single')}_{today}.mp4"

# ------------------------------------------------ 要处理的 PCD 列表
if args.sample:
    pcd_list = [args.sample]
else:
    pcd_list = sorted(glob.glob(os.path.join(args.scene_dir,'MergedPointCloud','*.pcd')))
    if not pcd_list:
        raise FileNotFoundError('在 scene_dir 里没找到任何 .pcd')

# ------------------------------------------------ 主循环
writer = None                       # video writer 延迟创建（拿到第一帧分辨率再建）
print(f'▶ Total frames: {len(pcd_list)}  –  Saving to {video_path}')
for idx, pcd_path in enumerate(pcd_list):
    batch = load_single_sample(pcd_path)          # 只传路径即可
    to_dev = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
    batch  = {k: to_dev(v) if not isinstance(v,dict)
                   else {kk:to_dev(vv) for kk,vv in v.items()}
              for k,v in batch.items()}

    # --- 前向
    out = model(batch['lidar_bev'], batch['front_img'], batch['side_imgs'], mode='test')
    traj = out['best_trajectories'][0].cpu().numpy()   # [40,3]

    # --- 取 BEV 第 5 通道做底图
    bev = batch['lidar_bev'][0,5].cpu().numpy()        # [H,W] 0~1
    H,W  = bev.shape
    x_rng=(-50,50); y_rng=(-50,50); grid=0.2           # 和 data_utils 保持一致

    # ---------- 绘图 ----------
    fig = plt.figure(figsize=(5,5), dpi=120)
    ax  = fig.add_subplot(111)
    ax.imshow(bev, cmap='gray', origin='lower',
              extent=[x_rng[0],x_rng[1],y_rng[0],y_rng[1]])
    ax.plot(traj[:,0], traj[:,1], 'r-', linewidth=2)
    ax.scatter(traj[-1,0], traj[-1,1], c='r', s=10)
    ax.set_xlim(*x_rng); ax.set_ylim(*y_rng); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{idx+1}/{len(pcd_list)}')

    # ---------- 保存帧 ----------
    frame_png = f'vis_out/frames/{idx:05d}.png'
    plt.savefig(frame_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ---------- 写入 video ----------
    img = iio.imread(frame_png)        # 读 png → ndarray
    if writer is None:                 # 首帧时创建 writer
        writer = iio.get_writer(video_path, fps=args.fps, codec='libx264')
    writer.append_data(img)

    print(f'  ✓ {idx+1:04d}/{len(pcd_list)}  {os.path.basename(pcd_path)}')

if writer: writer.close()
print(f'\n✅  已生成视频: {video_path}')