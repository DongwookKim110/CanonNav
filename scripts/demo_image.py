"""Single-image demo: raw camera image + goal -> SoR / path visualization.

python scripts/demo_image.py --image assets/sample_hallway.jpg --checkpoint checkpoints/canonnav.pt \
    --camera configs/camera/zed.yaml --goal 10.0 0.3 --out outputs/demo_goal.jpg
"""
import argparse
import os

import cv2
import torch

from canonnav.config import CameraCfg, load_config
from canonnav.inference import CanonNavPredictor
from canonnav.utils.visualize import visualize_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/canonnav.yaml")
    p.add_argument("--camera", default="configs/camera/zed.yaml")
    p.add_argument("--goal", type=float, nargs=2, metavar=("R", "THETA"), default=None,
                   help="goal in the robot frame: r [m], theta [rad, left +]. Omit for exploration.")
    p.add_argument("--sample-num", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/vis.jpg")
    p.add_argument("--device", default=None)
    p.add_argument("overrides", nargs="*", help="config dotlist overrides, e.g. network.dino.repo_dir=/x")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    camera = load_config(args.camera, schema=CameraCfg)
    predictor = CanonNavPredictor(cfg.network, args.checkpoint, camera, device=args.device)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)

    torch.manual_seed(args.seed)
    goal_directed = args.goal is not None
    goal_rt = tuple(args.goal) if goal_directed else (0.0, 0.0)
    res = predictor(bgr, goal_rt=goal_rt, goal_directed=goal_directed, sample_num=args.sample_num)

    vis = visualize_prediction(
        rgb=res.warped_rgb, K=predictor.K, camera_height=predictor.camera_height,
        sor_mu=res.sor_mu, sor_std=res.sor_std,
        paths=res.paths, best_index=res.best_index,
        goal_rt=goal_rt if goal_directed else None,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"SoR mu={res.sor_mu.round(2)} std={res.sor_std.round(2)} [m], best path #{res.best_index}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
