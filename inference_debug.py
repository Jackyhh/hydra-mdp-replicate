#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速推理-可视化脚本

* 支持 **SyntheticDataset** 与真实 Navsim / nuPlan 样本
* 仅依赖 perception_network 与 data_utils, 不修改原源码
* 运行后在 ./plots/debug 生成:
    - front_cam.png          - 输入前视图 (反归一化)
    - bev_segmentation.png   - BEV argmax 语义图
    - obj_confidence.png     - 置信度热力图
    - bbox_vis.png           - 在置信度热力图上叠加 bbox

用法
-----
::

    python inference_debug.py                      # SyntheticDataset 首个样本
    python inference_debug.py --ckpt checkpoints/best_model.pth --sample 42
    python inference_debug.py --real Navsim        # 真实数据集首样本
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from perception_network import build_perception_network
from data_utils import SyntheticDataset, create_dataloaders

# ---------- 常量/工具 ---------- #
COLORS = {
    "bbox": (0, 1, 0, 0.8),  # RGBA, matplotlib 0-1
}

PLOT_DIR = Path("./plots/debug")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """反归一化 [3,H,W] Tensor -> 0-1 之间"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def save_front_image(front_img: torch.Tensor):
    TF.to_pil_image(denormalize(front_img.cpu())).save(PLOT_DIR / "front_cam.png")


def save_bev_segmentation(seg_logits: torch.Tensor):
    seg_pred = torch.argmax(seg_logits, dim=0).cpu().numpy().astype(np.uint8)
    plt.figure(figsize=(4, 4))
    plt.title("BEV segmentation")
    plt.imshow(seg_pred, cmap="tab20")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "bev_segmentation.png", dpi=150)
    plt.close()


def save_obj_confidence(conf_map: torch.Tensor):
    plt.figure(figsize=(4, 4))
    plt.title("Obj-detect confidence")
    plt.imshow(conf_map.cpu().numpy(), cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "obj_confidence.png", dpi=150)
    plt.close()


def draw_bboxes_on_conf(conf_map: np.ndarray,
                        bbox_params: torch.Tensor,
                        thr: float = 0.5):
    """
    在置信度热图上绘制 bbox 轮廓 (粗略示意).

    conf_map     : [H,W] numpy
    bbox_params  : [7,H,W] tensor  (x,y,z,w,l,h,theta)
    """
    H, _ = conf_map.shape
    grid_size = 100.0 / H  # 近似每像素对应米数 (假设场景宽 100m)

    plt.figure(figsize=(6, 6))
    plt.title("bbox_vis")
    plt.imshow(conf_map, cmap="hot")

    xs, ys = np.where(conf_map > thr)  # row,col
    for row, col in zip(xs, ys):
        w = float(bbox_params[3, row, col]) / grid_size
        l = float(bbox_params[4, row, col]) / grid_size
        if w <= 0 or l <= 0:
            continue

        cx, cy = col, row  # matplotlib 坐标：x=col, y=row
        theta = float(bbox_params[6, row, col])

        half_w, half_l = w / 2, l / 2
        corners = np.array(
            [[ half_l,  half_w],
             [ half_l, -half_w],
             [-half_l, -half_w],
             [-half_l,  half_w]], dtype=np.float32
        )
        rot = np.array([[ math.cos(theta), -math.sin(theta)],
                        [ math.sin(theta),  math.cos(theta)]], dtype=np.float32)
        corners = corners @ rot.T
        corners[:, 0] += cx  # 累加偏移
        corners[:, 1] += cy

        poly = plt.Polygon(corners, fill=False,
                           edgecolor=COLORS["bbox"], linewidth=0.8)
        plt.gca().add_patch(poly)
        plt.text(cx, cy, f"{conf_map[row, col]:.2f}",
                 color="white", fontsize=4, ha="center", va="center")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "bbox_vis.png", dpi=150)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser("Inference debug for PerceptionNetwork")
    ap.add_argument("--ckpt",   type=str, default="checkpoints/latest_checkpoint.pth",
                    help="checkpoint path (.pth)")
    ap.add_argument("--sample", type=int, default=0,
                    help="sample index to visualize (Synthetic)")
    ap.add_argument("--real",   default="Navsim",
                    choices=["", "Navsim", "nuPlan"],
                    help="use real dataset instead of Synthetic")
    return ap.parse_args()


def main():
    args = parse_args()

    # 支持 CUDA → MPS → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # 1. 构建并加载模型
    model = build_perception_network().to(device).eval()
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device)
        miss, unexp = model.load_state_dict(
            ckpt.get("model_state_dict", ckpt),
            strict=False
        )
        print(f"Loaded ckpt {args.ckpt}. missing={len(miss)}, unexpected={len(unexp)}")

    # 2. 准备数据
    if args.real:

        # from pprint import pprint           # 放到文件顶部已 import 的下面也可
        # from config import PERCEPTION_CONFIG

        # print(">> PERCEPTION_CONFIG 内容预览:")
        # pprint(PERCEPTION_CONFIG)           # 一次性把字典结构完整打印出来

        # # —— 在这里插入 —— 
        # # 仅保留对 PERCEPTION_CONFIG 的引用，Path 直接用全局的
        # from config import PERCEPTION_CONFIG   # 已有

        # data_dir = Path(PERCEPTION_CONFIG.get('DATA_DIR', 
        #             PERCEPTION_CONFIG.get('data_root', '/tmp/unknown')))
        # print(">> real data dir:", data_dir)

        # cam_dir = Path(data_dir) / "images" / "CAM_F0"
        # print(">> 头几张 CAM_F0 文件:", sorted(cam_dir.glob("*.png"))[:3])
        # # —— 插入结束 ——
        
        from config import PERCEPTION_CONFIG, DATA_DIR
        data_dir = Path(DATA_DIR)
        print(">> PERCEPTION_CONFIG:", PERCEPTION_CONFIG)
        print(">> real data dir:", data_dir)
        cam_dir = data_dir / "images" / "CAM_F0"
        print(">> 头几张 CAM_F0 文件:", sorted(cam_dir.glob('*.png'))[:3])
        # -----------------------------------------


        _, _, test_loader = create_dataloaders(
            batch_size=1, num_workers=0, distributed=False
        )
        sample = next(iter(test_loader))
        lidar     = sample["lidar"][0]
        front_img = sample["images"]["CAM_F0"][0]
        side_imgs = None


    else:
        dataset = SyntheticDataset(num_samples = max(1, args.sample + 1))
        sample  = dataset[args.sample]
        lidar     = sample["lidar"]
        front_img = sample["images"]["CAM_F0"]
        side_imgs = None

    # 3. 推理
    with torch.no_grad():
        outputs = model(
            lidar.unsqueeze(0).to(device),
            front_img.unsqueeze(0).to(device),
            side_imgs
        )

    # 4. 保存可视化
    if front_img.abs().sum() < 1e-6:
        print("⚠️  CAM_F0 为空图像，可能是占位符。")
    else:
        save_front_image(front_img)

    seg_logits = outputs["bev_segmentation"][0].cpu()
    save_bev_segmentation(seg_logits)

    obj_map  = outputs["obj_detection"][0].cpu()  # [C,H,W]
    # conf_map = obj_map[7]                        # 置信度通道
    conf_map = torch.sigmoid(obj_map[7])   # 保证是 0~1
    print('conf_map.max: ', conf_map.max())

    save_obj_confidence(conf_map)

    # 3.1 统计阈值以上的 bounding-box 位置
    thr = 0.5              # 你也可以改成 0.3 / 0.6 做对比
    mask = (conf_map > thr)
    num_preds = int(mask.sum())
    max_conf  = float(conf_map.max())
    print(f"[bbox] 置信度>{thr} 的格子数 = {num_preds} （最高 {max_conf:.2f}）")

    obj_map = outputs["obj_detection"][0].cpu()   # [C, H, W]
    print("obj_map.shape =", obj_map.shape)       # ← C 应该=8(7参数+conf) 或你实现的通道数
    print("max per-channel:", obj_map.view(obj_map.shape[0], -1).max(dim=1).values)



    # >>>★★ 这里插入打印 ★★<<<
    for thr in [0.1, 0.05, 0.01]:
        n = int((conf_map > thr).sum())
        print(f"thr={thr:.2f}  置信格子数={n},  max={float(conf_map.max()):.4f}")
    # >>>★★ 打印结束 ★★<<<


    draw_bboxes_on_conf(conf_map.numpy(), obj_map[:7])

    
    overlay_bboxes_on_rgb(front_img.cpu(), conf_map.numpy(), obj_map[:7], thr=0.5)

    print("Saved visualizations to", PLOT_DIR.resolve())




def overlay_bboxes_on_rgb(rgb_img, conf_map, bbox_params, thr=0.5):
    """
    把 BEV 检测的 bbox 粗略投射到 2D 图像（简化：直接画矩形块示意）
    仅用于直观验证；并非真正相机外参变换
    """
    import cv2
    h, w = rgb_img.shape[1:]   # pytorch CHW
    rgb = denormalize(rgb_img).permute(1, 2, 0).numpy() * 255
    rgb = rgb.astype(np.uint8).copy()

    xs, ys = np.where(conf_map > thr)
    for row, col in zip(xs, ys):
        # 直接用格子划分 -> 映射到图上大致方块，方便看看数量
        x0 = int(col * w / conf_map.shape[1])
        y0 = int(row * h / conf_map.shape[0])
        x1 = int((col+1) * w / conf_map.shape[1])
        y1 = int((row+1) * h / conf_map.shape[0])
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (0,255,0), 1)

    cv2.imwrite(str(PLOT_DIR / "front_cam_bbox.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("→ 已保存 front_cam_bbox.png")


if __name__ == "__main__":
    main()