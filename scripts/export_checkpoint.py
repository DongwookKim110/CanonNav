"""Convert a training checkpoint into a release checkpoint (EMA weights, LoRA adapters only).

python scripts/export_checkpoint.py <train_ckpt.pt> checkpoints/canonnav.pt
"""
import argparse

from canonnav.models.build import export_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train_ckpt")
    p.add_argument("out")
    args = p.parse_args()
    export_checkpoint(args.train_ckpt, args.out)
    print(f"saved: {args.out}")
