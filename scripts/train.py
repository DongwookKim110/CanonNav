"""Train CanonNav.

single GPU:  python scripts/train.py --config configs/train.yaml
DDP:         torchrun --nproc_per_node=4 scripts/train.py --config configs/train.yaml
overrides:   python scripts/train.py --config configs/train.yaml train.batch_size=32 train.output_dir=runs/test
"""
import argparse

from canonnav.config import TrainConfig, load_config
from canonnav.engine.distributed import cleanup_distributed
from canonnav.engine.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("overrides", nargs="*", help="config dotlist overrides, e.g. train.epoch=10")
    args = p.parse_args()

    trainer = Trainer(load_config(args.config, args.overrides, schema=TrainConfig))
    try:
        trainer.train()
    finally:
        cleanup_distributed(trainer.dist)


if __name__ == "__main__":
    main()
