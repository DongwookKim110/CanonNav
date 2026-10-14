"""Generate training labels for every sequence under a dataset root.

Layout:  <root>/<env>/calib.yaml,  <root>/<env>/<seq>/{rgb/, trav_map/, odometry.csv}
Output:  <root>/<env>/<seq>/labels/<stamp>.npz   (+ labels_vis/<stamp>.jpg with --vis)

python scripts/preprocess.py --root /path/to/dataset [--vis]
"""
import argparse
from pathlib import Path

from tqdm import tqdm

from canonnav.config import PreprocessCfg, load_config
from canonnav.data.preprocess import find_sequences, load_calib, process_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root (or a single environment / sequence folder)")
    p.add_argument("--config", default="configs/preprocess.yaml")
    p.add_argument("--vis", action="store_true", help="also write label visualizations")
    p.add_argument("overrides", nargs="*", help="config dotlist overrides, e.g. output.label_folder=labels_v2")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides, schema=PreprocessCfg)
    seqs = find_sequences(args.root, cfg)
    if not seqs:
        raise SystemExit(f"no sequences (folders containing '{cfg.dataset.rgb_folder}/') under {args.root}")

    total_saved = total_frames = 0
    for seq in seqs:
        calib = load_calib(seq.parent / cfg.dataset.calib_file)
        saved, frames = process_sequence(seq, calib, cfg, vis=args.vis,
                                         progress=lambda it: tqdm(it, desc=str(seq), leave=False))
        print(f"{seq}: {saved} labels from {frames} frames")
        total_saved += saved
        total_frames += frames
    print(f"total: {total_saved} labels from {total_frames} frames in {len(seqs)} sequences")


if __name__ == "__main__":
    main()
