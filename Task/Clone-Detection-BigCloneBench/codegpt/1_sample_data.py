#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from collections import Counter

# python sample_data.py --seed 3 --mode balanced

def read_lines(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    # drop empty lines just in case
    return [ln for ln in lines if ln.strip()]


def get_label(line: str) -> str:
    # expecting: url1 \t url2 \t label
    parts = line.split("\t")
    if len(parts) < 3:
        raise ValueError(f"Malformed line (expected 3 TSV fields): {line[:120]}")
    return parts[2].strip()  # "0" or "1"


def write_lines(out_path: Path, lines):
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stratified_sample(lines, ratio: float, rnd: random.Random):
    by_lbl = {"0": [], "1": []}
    for ln in lines:
        by_lbl[get_label(ln)].append(ln)

    out = []
    for lbl, group in by_lbl.items():
        k = int(len(group) * ratio)
        if k > 0:
            out.extend(rnd.sample(group, k))
    rnd.shuffle(out)
    return out


def balanced_sample(lines, ratio: float, rnd: random.Random):
    """
    Make a balanced (50:50) dataset WITHOUT oversampling.
    Sample size will be limited by the minority class size.
    ratio is interpreted as a cap relative to the full dataset size,
    but balance requirement may reduce it further.
    """
    by_lbl = {"0": [], "1": []}
    for ln in lines:
        by_lbl[get_label(ln)].append(ln)

    total_target = int(len(lines) * ratio)
    # want half/half
    half_target = total_target // 2

    # cannot exceed minority size
    max_half = min(len(by_lbl["0"]), len(by_lbl["1"]))
    half = min(half_target, max_half)

    out = rnd.sample(by_lbl["0"], half) + rnd.sample(by_lbl["1"], half)
    rnd.shuffle(out)
    return out


def balanced_oversample(lines, ratio: float, rnd: random.Random):
    """
    Make a balanced (50:50) dataset WITH oversampling to hit ratio size.
    total size = int(N*ratio). If minority is too small, sample with replacement.
    """
    by_lbl = {"0": [], "1": []}
    for ln in lines:
        by_lbl[get_label(ln)].append(ln)

    total_target = int(len(lines) * ratio)
    half = total_target // 2

    maj = "0" if len(by_lbl["0"]) >= len(by_lbl["1"]) else "1"
    mino = "1" if maj == "0" else "0"

    # majority: sample without replacement if possible, else with replacement (unlikely)
    if len(by_lbl[maj]) >= half:
        maj_part = rnd.sample(by_lbl[maj], half)
    else:
        maj_part = [rnd.choice(by_lbl[maj]) for _ in range(half)]

    # minority: oversample if needed
    if len(by_lbl[mino]) >= half:
        min_part = rnd.sample(by_lbl[mino], half)
    else:
        min_part = [rnd.choice(by_lbl[mino]) for _ in range(half)]

    out = maj_part + min_part
    rnd.shuffle(out)
    return out


def summarize(name, lines):
    cnt = Counter(get_label(ln) for ln in lines)
    total = len(lines)
    print(f"{name}: total={total} label0={cnt.get('0',0)} label1={cnt.get('1',0)}")


def process_one(in_path: Path, out_path: Path, ratio: float, seed: int, mode: str):
    rnd = random.Random(seed)
    lines = read_lines(in_path)
    summarize(f"[IN ] {in_path.name}", lines)

    if mode == "stratified":
        out = stratified_sample(lines, ratio, rnd)
    elif mode == "balanced":
        out = balanced_sample(lines, ratio, rnd)
    elif mode == "balanced_oversample":
        out = balanced_oversample(lines, ratio, rnd)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    write_lines(out_path, out)
    summarize(f"[OUT] {out_path.name}", out)
    print(f"[OK] {in_path} -> {out_path} (mode={mode}, ratio={ratio}, seed={seed})\n")


def main():
    p = argparse.ArgumentParser(description="Create deterministic sampled train/valid files.")
    p.add_argument("--dataset_dir", type=str, default="../dataset")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--mode", type=str, default="stratified",
                   choices=["stratified", "balanced", "balanced_oversample"],
                   help="stratified keeps original label ratio; balanced makes 50:50; balanced_oversample makes 50:50 with duplication.")
    args = p.parse_args()

    d = Path(args.dataset_dir)
    process_one(d/"train.txt", d/"train_10percent.txt", args.ratio, args.seed, args.mode)
    process_one(d/"valid.txt", d/"valid_10percent.txt", args.ratio, args.seed, args.mode)


if __name__ == "__main__":
    main()