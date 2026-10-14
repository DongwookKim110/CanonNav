"""Typed configuration schema.

Configs are parsed once at the entry point into dataclass objects; builders only
receive typed sub-configs (e.g. `cfg.network`), never raw dicts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, TypeVar

from omegaconf import OmegaConf

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
@dataclass
class DinoCfg:
    repo_dir: str = "???"   # local clone of https://github.com/facebookresearch/dinov3
    model: str = "dinov3_vits16plus"
    pth: str = "???"        # pretrained DINOv3 weight file


@dataclass
class DecoderCfg:
    """Goal-conditioned cross-attention decoder."""
    nhead: int = 8
    cross_attn_layers: int = 2
    mlp_ratio: int = 2
    out_channel: int = 256
    goal_r_max: float = 60.0


@dataclass
class DiffusionPolicyCfg:
    pred_action_step: int = 16
    diffusion_step: int = 10


@dataclass
class SorHeadCfg:
    hidden_dim: int = 256


@dataclass
class PathTraversabilityNetCfg:
    hidden: int = 256
    ch: int = 256


@dataclass
class NetworkCfg:
    # model input resolution and the canonical intrinsic every image is warped to
    W: int = 336
    H: int = 256
    # 3x3 intrinsic. Nested list, so typed as Any to skip OmegaConf strict validation.
    K: Any = field(default_factory=lambda: [[168.0, 0.0, 168.0],
                                            [0.0, 168.0, 128.0],
                                            [0.0, 0.0, 1.0]])
    dino: DinoCfg = field(default_factory=DinoCfg)
    decoder: DecoderCfg = field(default_factory=DecoderCfg)
    diffusion_policy: DiffusionPolicyCfg = field(default_factory=DiffusionPolicyCfg)
    lora_rank: int = 3
    sor_head: SorHeadCfg = field(default_factory=SorHeadCfg)
    path_traversability_net: PathTraversabilityNetCfg = field(default_factory=PathTraversabilityNetCfg)


@dataclass
class Config:
    network: NetworkCfg = field(default_factory=NetworkCfg)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class DataSourceCfg:
    """One environment folder (every sequence below it is used) and its sampling weight."""
    path: str = "???"
    weight: float = 1.0


@dataclass
class DataCfg:
    train: List[DataSourceCfg] = field(default_factory=list)
    # validation sequences (or folders containing them); they are not removed from training
    val: List[str] = field(default_factory=list)
    rgb_folder: str = "rgb"
    label_folder: str = "labels"


@dataclass
class BevCfg:
    """BEV grid of the labels (must match configs/preprocess.yaml)."""
    cell_length: float = 0.1
    horizontal_range: List[float] = field(default_factory=lambda: [-32.0, 32.0])
    forward_range: List[float] = field(default_factory=lambda: [0.0, 32.0])


@dataclass
class SupervisionCfg:
    robot_size: float = 0.25           # clearance below which a waypoint counts as unsafe [m]
    safety_margin: float = 0.5         # extra clearance the collision loss asks for [m]
    path_consistency_tau: float = 0.5  # soft-min temperature of the SoR-path consistency loss


@dataclass
class LrCfg:
    lora: float = 5e-5
    side_branch: float = 1e-4
    goal_conditioned_attn_layer: float = 1e-4
    diffuser_noise_pred_net: float = 1e-4
    sor_head: float = 1e-4
    path_traversability_net: float = 1e-4


@dataclass
class LossWeightCfg:
    sor_nll: float = 2.0
    sor_neg: float = 8.0
    sor_path_consistency: float = 0.25
    diffusion: float = 25.0
    traversability: float = 5.0
    collision: float = 0.3
    curvature: float = 0.05


@dataclass
class ColorJitterCfg:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    p: float = 0.8


@dataclass
class TrainCfg:
    gpu_ids: Optional[List[int]] = None     # None: all visible GPUs; DDP maps local_rank -> gpu_ids[local_rank]
    seed: int = 0
    epoch: int = 70
    batch_size: int = 64                    # per GPU
    path_per_image: int = 1                 # goals sampled per image
    exploration_prob: float = 0.3           # chance that a sample is an exploration (no-goal) sample
    rotation_aug: float = 15.0              # random yaw augmentation range [deg]
    color_jitter: ColorJitterCfg = field(default_factory=ColorJitterCfg)
    lr: LrCfg = field(default_factory=LrCfg)
    loss_weight: LossWeightCfg = field(default_factory=LossWeightCfg)
    val_every: int = 10                    # validate every N epochs
    val_vis_frames: int = 20                # frames per validation round saved as images
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    output_dir: str = "???"                 # checkpoints, log, validation images
    resume: Optional[str] = None            # training checkpoint to continue from


@dataclass
class TrainConfig:
    network: NetworkCfg = field(default_factory=NetworkCfg)
    bev: BevCfg = field(default_factory=BevCfg)
    supervision: SupervisionCfg = field(default_factory=SupervisionCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


# --------------------------------------------------------------------------- #
# camera (inference input)
# --------------------------------------------------------------------------- #
@dataclass
class CameraCfg:
    """Physical camera that produced the input image."""
    K: Any = "???"             # 3x3 intrinsic of the raw image
    size: List[int] = "???"    # [H, W] of the raw image
    height: float = "???"      # camera height above ground [m]
    pitch: float = 0.0         # [deg], + looks up (sky), - looks down (ground); as in calib.yaml


# --------------------------------------------------------------------------- #
# data preprocessing (label generation)
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessDatasetCfg:
    """Layout: <root>/<env>/<calib_file>, <root>/<env>/<seq>/{<rgb_folder>, <trav_map_folder>, <odometry_file>}"""
    rgb_folder: str = "rgb"
    trav_map_folder: str = "trav_map"   # traversability mask per frame: 255 traversable, 0 not
    odometry_file: str = "odometry.csv"
    calib_file: str = "calib.yaml"


@dataclass
class PreprocessBevCfg:
    cell_size: float = 0.1
    horizontal_range: List[float] = field(default_factory=lambda: [-32.0, 32.0])   # left [m]
    forward_range: List[float] = field(default_factory=lambda: [0.0, 32.0])        # forward [m]


@dataclass
class PreprocessTrajCfg:
    interp_step_m: float = 0.33          # future trajectory resampled at this spacing [m]
    min_traj_len: int = 2                # frames with fewer future poses are skipped
    safety_ratio_threshold: float = 0.7  # min free fraction of the robot footprint on the BEV


@dataclass
class PreprocessOutputCfg:
    label_folder: str = "labels"         # <seq>/<label_folder>/<stamp>.npz
    vis_folder: str = "labels_vis"       # <seq>/<vis_folder>/<stamp>.jpg (with --vis)


@dataclass
class PreprocessCfg:
    dataset: PreprocessDatasetCfg = field(default_factory=PreprocessDatasetCfg)
    bev: PreprocessBevCfg = field(default_factory=PreprocessBevCfg)
    traj: PreprocessTrajCfg = field(default_factory=PreprocessTrajCfg)
    output: PreprocessOutputCfg = field(default_factory=PreprocessOutputCfg)


def _load_yaml(path: str):
    """Load a yaml file; a top-level `base: <file>` (relative to this file) is loaded first and overridden."""
    cfg = OmegaConf.load(path)
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    return OmegaConf.merge(_load_yaml(os.path.join(os.path.dirname(path), base)), cfg)


def load_config(path: str, overrides: Optional[List[str]] = None, schema: Type[T] = Config) -> T:
    """Merge a yaml file (and its bases, and optional dotlist overrides) into `schema` and return the dataclass."""
    cfg = OmegaConf.merge(OmegaConf.structured(schema), _load_yaml(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]
