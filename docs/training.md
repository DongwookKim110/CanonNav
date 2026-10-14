# Training

Run all commands from the repository root with the `canonnav` environment activated
(see [Installation](../README.md#installation)).

- [1. Prepare the dataset](#1-prepare-the-dataset)
- [2. Generate labels](#2-generate-labels)
- [3. Train](#3-train)
- [4. Export a checkpoint](#4-export-a-checkpoint)

## 1. Prepare the dataset

Download the [sample dataset](https://github.com/DongwookKim110/CanonNav/releases/download/v1.0/canonnav_sample_dataset.zip) (654 MB; campus, offroad, parking) and unzip it:

```bash
wget https://github.com/DongwookKim110/CanonNav/releases/download/v1.0/canonnav_sample_dataset.zip
unzip canonnav_sample_dataset.zip   # -> sample_dataset/
```

The sample dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Each environment folder holds one `calib.yaml` and one or more recorded sequences:

```
<root>/<env>/calib.yaml                   camera intrinsics/extrinsics, robot footprint, sampling settings
<root>/<env>/<seq>/rgb/<stamp>.jpg         camera images
<root>/<env>/<seq>/trav_map/<stamp>.png    traversability mask per image (255 traversable, 0 not)
<root>/<env>/<seq>/odometry.csv            robot pose per image (one row per image, same order)
```

To train on your own data, record it in the same layout; the files of the sample dataset show the
`calib.yaml` and `odometry.csv` formats.

> **Note:** `rgb/`, `trav_map/` and `odometry.csv` are paired by order, so they must have the same number of
> entries in the same order.

## 2. Generate labels

```bash
python scripts/preprocess.py --root /path/to/sample_dataset --vis
```

For each sequence this writes `<seq>/labels/<stamp>.npz` (goals, SoR targets, future trajectory, BEV), and with
`--vis` also `<seq>/labels_vis/<stamp>.jpg` for checking them. Settings are in `configs/preprocess.yaml`.
Training reads only the labels, so run this again after changing a `calib.yaml`.

## 3. Train

Set the dataset paths in `configs/train.yaml` (the network and the DINOv3 paths come from `configs/canonnav.yaml`):

```yaml
data:
  train:                                  # environment folders and their sampling weights
    - { path: /path/to/sample_dataset/campus, weight: 1.0 }
    - { path: /path/to/sample_dataset/offroad, weight: 1.0 }
  val:                                    # sequences used for validation
    - /path/to/sample_dataset/campus/seq2
```

```bash
# release setting: 4 GPUs (DDP), 70 epochs
torchrun --nproc_per_node=4 scripts/train.py --config configs/train.yaml

# quick check on one GPU
python scripts/train.py --config configs/train.yaml \
    train.epoch=2 train.batch_size=8 train.val_every=1 train.output_dir=runs/test

# resume
python scripts/train.py --config configs/train.yaml train.resume=runs/canonnav/checkpoints/010.pt
```

Any config value can be overridden at the end of the command as above. Outputs in `train.output_dir`
(default `runs/canonnav`):

```
checkpoints/<epoch>.pt   model, EMA weights and optimizer state (LoRA adapters only for the backbone)
train_log.txt            train / validation losses
val/<epoch>/*.jpg        predictions on the validation set
config.yaml              the resolved config
```

> **Note (difference from the paper):** this code also trains an exploration mode with goal masking, as in [NoMaD](https://arxiv.org/abs/2310.07896): the goal of each sample is masked with probability 0.3 (`train.exploration_prob`), so the same model also runs without a goal.

## 4. Export a checkpoint

Training checkpoints can be used directly by the demo and the ROS node (their EMA weights are used).
To make a smaller release checkpoint (~80 MB):

```bash
python scripts/export_checkpoint.py runs/canonnav/checkpoints/069.pt checkpoints/canonnav.pt
```
