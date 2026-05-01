#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List

DEFAULT_SEED = 3


def prefix_pairs(in_path: Path, out_path: Path, prefix: str):
    """
    Prefix pair identifiers in a clone-pair TSV file to avoid ID collisions.
    Supports both tab- and space-separated files.
    Format: <id1> <id2> <label>
    Output is normalized as TSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                skipped += 1
                continue
            u1, u2, y = parts
            fout.write(f"{prefix}{u1}\t{prefix}{u2}\t{y}\n")
            n += 1
    print(f"[OK] prefixed pairs: {in_path} -> {out_path} (lines={n}, skipped={skipped})")


def build_mix_jsonl(bcb_jsonl: Path, other_jsonl: Path, out_jsonl: Path, other_prefix: str):
    """
    Build combined mapping jsonl:
      - BCB entries prefixed with 'bcb_'
      - Other-domain entries prefixed with '<otherdomain>_'
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()

    def append_with_prefix(src: Path, prefix: str):
        n = 0
        with src.open("r", encoding="utf-8") as fin, out_jsonl.open("a", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                if "idx" not in js:
                    continue
                js["idx"] = f"{prefix}{js['idx']}"
                fout.write(json.dumps(js, ensure_ascii=False) + "\n")
                n += 1
        print(f"[OK] appended jsonl: {src} (+{n}) with prefix={prefix}")

    append_with_prefix(bcb_jsonl, "bcb_")
    append_with_prefix(other_jsonl, other_prefix)


def merge_shuffle(out_path: Path, files: List[Path], seed: int):
    rnd = random.Random(seed)
    lines: List[str] = []
    for f in files:
        with Path(f).open("r", encoding="utf-8") as fin:
            lines.extend([ln for ln in fin if ln.strip()])
    rnd.shuffle(lines)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] merged+shuffled: {out_path} (total_lines={len(lines)})")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Create mixed (BCB + other-domain) train/valid datasets and combined mix/data.jsonl. "
            "Optionally generate an OTHER-DOMAIN-ONLY test file with prefixed IDs at "
            "../dataset/test_<otherdomain_name>.txt (no overwrite of ../dataset/test.txt)."
        )
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Other domain name (controls prefix + output test filename)
    p.add_argument("--otherdomain_name", type=str, required=True,
                   help="Other domain name used as prefix and test output naming, e.g., camel -> camel_ and test_camel.txt")

    # Required pair files
    p.add_argument("--train_data_file_bcb", type=str, required=True)
    p.add_argument("--valid_data_file_bcb", type=str, required=True)
    p.add_argument("--train_data_file_more", type=str, required=True)
    p.add_argument("--valid_data_file_more", type=str, required=True)

    # Optional: other-domain test file (other-domain-only evaluation)
    p.add_argument("--test_data_file_otherdomain", type=str, default=None,
                   help="Other-domain test pairs file (id1 id2 label). If provided, generates ../dataset/test_<otherdomain_name>.txt")

    # Mapping files
    p.add_argument("--bcb_jsonl", type=str, required=True)
    p.add_argument("--more_jsonl", type=str, required=True)

    # Outputs
    p.add_argument("--out_dir", type=str, default="../dataset/mix")
    p.add_argument("--out_train_mix", type=str, default="../dataset/train_mix.txt")
    p.add_argument("--out_valid_mix", type=str, default="../dataset/valid_mix.txt")

    args = p.parse_args()

    other_prefix = f"{args.otherdomain_name}_"
    print(f"[INFO] other-domain prefix = {other_prefix}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prefix train/valid
    bcb_train_p = out_dir / "train_bcb.pref.txt"
    bcb_valid_p = out_dir / "valid_bcb.pref.txt"
    other_train_p = out_dir / "train_other.pref.txt"
    other_valid_p = out_dir / "valid_other.pref.txt"

    prefix_pairs(Path(args.train_data_file_bcb), bcb_train_p, "bcb_")
    prefix_pairs(Path(args.valid_data_file_bcb), bcb_valid_p, "bcb_")
    prefix_pairs(Path(args.train_data_file_more), other_train_p, other_prefix)
    prefix_pairs(Path(args.valid_data_file_more), other_valid_p, other_prefix)

    # 2) Build combined jsonl
    mix_jsonl = out_dir / "data.jsonl"
    build_mix_jsonl(Path(args.bcb_jsonl), Path(args.more_jsonl), mix_jsonl, other_prefix)

    # 3) Merge train / valid
    merge_shuffle(Path(args.out_train_mix), [bcb_train_p, other_train_p], seed=args.seed)
    merge_shuffle(Path(args.out_valid_mix), [bcb_valid_p, other_valid_p], seed=args.seed)

    # 4) Generate OTHER-DOMAIN-ONLY test file (no overwrite)
    if args.test_data_file_otherdomain:
        out_test = Path("../dataset") / f"test_{args.otherdomain_name}.txt"
        prefix_pairs(Path(args.test_data_file_otherdomain), out_test, other_prefix)
        print(f"[OK] wrote OTHER-DOMAIN-ONLY test file: {out_test}")
    else:
        print("[SKIP] No other-domain test file requested.")

    print("\nDone.")
    print(f"- train_mix: {args.out_train_mix}")
    print(f"- valid_mix: {args.out_valid_mix}")
    print(f"- mix jsonl:  {mix_jsonl}")
    if args.test_data_file_otherdomain:
        print(f"- test(other-domain only): ../dataset/test_{args.otherdomain_name}.txt")


if __name__ == "__main__":
    main()
