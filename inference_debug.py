#/home/jiaqi/workspace/code/hydra-mdp/date20250711/pythonProject/inference_debug.py
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
    # ------------↑ 修正：漏写 “+” --------------
    return (img * std + mean).clamp(0, 1)


def save_front_image(front_img: torch.Tensor):
    TF.to_pil_image(denormalize(front_img.cpu())).save(PLOT_DIR / "front_cam.png")


def save_bev_segmentation(seg_logits: torch.Tensor):
    seg_pred = torch.argmax(seg_logits, dim=0).numpy().astype(np.uint8)
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
    plt.imshow(conf_map, cmap="hot")
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
        # -----------↑ corners 坐标旋转
        corners[:, 0] += cx   # ← 修正：累加而非赋值
        corners[:, 1] += cy   # ← 修正：累加而非赋值

        poly = plt.Polygon(corners, fill=False,
                           edgecolor=COLORS["bbox"], linewidth=0.8)
        plt.gca().add_patch(poly)
        plt.text(cx, cy, f"{conf_map[row, col]:.2f}",
                 color="white", fontsize=4, ha="center", va="center")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "bbox_vis.png", dpi=150)
    plt.close()


# ---------- 主程序 ---------- #
def parse_args():
    ap = argparse.ArgumentParser("Inference debug for PerceptionNetwork")
    ap.add_argument("--ckpt",   type=str, default="checkpoints/best_model.pth",
                    help="checkpoint path (.pth)")
    ap.add_argument("--sample", type=int, default=0,
                    help="sample index to visualize (Synthetic)")
    ap.add_argument("--real", default='Navsim',  choices=["", "Navsim", "nuPlan"],
                    help="use real dataset instead of Synthetic")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # 1. 构建模型
    model = build_perception_network().to(device).eval()

    # 2. 加载权重（若存在）
    if args.ckpt and Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location=device)
        miss, unexp = model.load_state_dict(ckpt.get("model_state_dict", ckpt),
                                            strict=False)
        print(f"Loaded ckpt {args.ckpt}. missing={len(miss)}, unexpected={len(unexp)}")

    # 3. 准备数据
    if args.real:
        # 真实数据：取 test_loader 第一帧
        _, _, test_loader = create_dataloaders(batch_size=1,
                                               num_workers=0,
                                               distributed=False)
        sample = next(iter(test_loader))
        lidar      = sample["lidar"][0]
        front_img  = sample["images"]["CAM_F0"][0]
        side_imgs  = None
    else:
        # Synthetic
        dataset = SyntheticDataset(num_samples=max(1, args.sample + 1))  # ← 修正
        sample  = dataset[args.sample]
        lidar      = sample["lidar"]
        front_img  = sample["images"]["CAM_F0"]
        side_imgs  = None

    # 4. 前向推理
    with torch.no_grad():
        outputs = model(lidar.unsqueeze(0).to(device),
                        front_img.unsqueeze(0).to(device),
                        side_imgs)

    # 5. 可视化输出
    if front_img.abs().sum() < 1e-6:
        print("⚠️  CAM_F0 为空图像（占位符）。请检查数据或换一帧。")
    else:
        save_front_image(front_img)

    seg_logits = outputs["bev_segmentation"][0].cpu()
    save_bev_segmentation(seg_logits)

    obj_map   = outputs["obj_detection"][0].cpu()  # [C,H,W]
    conf_map  = obj_map[7]                         # 置信度通道
    save_obj_confidence(conf_map)

    draw_bboxes_on_conf(conf_map.numpy(), obj_map[:7])

    print("Saved visualizations to", PLOT_DIR.resolve())


if __name__ == "__main__":
    main()

