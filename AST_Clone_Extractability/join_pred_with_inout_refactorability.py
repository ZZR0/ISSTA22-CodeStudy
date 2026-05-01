#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# func_id -> (in_count, out_count, in_types, out_types, out_vars, extractable)
FuncMeta = Tuple[int, int, List[str], List[str], List[str], Optional[bool]]


def _as_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    return []


def _as_sorted_unique_list(x: Any) -> List[str]:
    vals = [s.strip() for s in _as_list(x) if str(s).strip() != ""]
    return sorted(set(vals))


def build_func_map(jsonl_path: Path) -> Dict[str, FuncMeta]:
    func_map: Dict[str, FuncMeta] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            for s in obj.get("sources", []):
                fid = s.get("func_id")
                if not fid:
                    continue

                ins = _as_list(s.get("In", []))
                outs = _as_list(s.get("Out", []))
                in_types = _as_sorted_unique_list(s.get("InType", []))
                out_types = _as_sorted_unique_list(s.get("OutType", []))
                ext = s.get("Extractable", None)

                func_map[fid] = (
                    len(ins),
                    len(outs),
                    in_types,
                    out_types,
                    outs,          # needed for emptiness check
                    ext,
                )

    return func_map


def parse_pred_line(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def bool_to_str(x: Optional[bool]) -> str:
    if x is True:
        return "True"
    if x is False:
        return "False"
    return ""


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_pair_type_compatibility(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, bool]:
    """
    Implements:

      InTypeOK(i)  <=> inType(i_L) = inType(i_R)

      OutTypeOK(i) <=>
            (Out(i_L)=∅ and Out(i_R)=∅)
         OR (Out(i_L)≠∅ and Out(i_R)≠∅ and outType(i_L)=outType(i_R))

      TypeOK(i) <=> InTypeOK(i) ∧ OutTypeOK(i)
    """
    L_in_type = set(left.get("InType", []))
    R_in_type = set(right.get("InType", []))
    in_type_ok = (L_in_type == R_in_type)

    L_out = set(left.get("Out", []))
    R_out = set(right.get("Out", []))

    L_out_type = set(left.get("OutType", []))
    R_out_type = set(right.get("OutType", []))

    if not L_out and not R_out:
        out_type_ok = True
    elif L_out and R_out:
        out_type_ok = (L_out_type == R_out_type)
    else:
        out_type_ok = False

    return {
        "pair_inType_ok": in_type_ok,
        "pair_outType_ok": out_type_ok,
        "pair_type_ok": (in_type_ok and out_type_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Join clone predictions with function-level extractability and pair type compatibility."
    )
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: --jsonl not found: {args.jsonl}", file=sys.stderr)
        return 2
    if not args.pred.exists():
        print(f"ERROR: --pred not found: {args.pred}", file=sys.stderr)
        return 2

    func_map = build_func_map(args.jsonl)

    total_pairs = 0
    clone_ones = 0
    pair_refactorable_true = 0
    pair_type_ok_true = 0

    ensure_parent(args.out)

    with args.pred.open("r", encoding="utf-8") as fin, \
         args.out.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow([
            "pair_left_func",
            "pair_right_func",
            "clone_predict",
            "pair_left_in",
            "pair_right_in",
            "pair_left_out",
            "pair_right_out",
            "pair_inType_ok",
            "pair_outType_ok",
            "pair_type_ok",
            "pair_left_extractable",
            "pair_right_extractable",
            "pair_refactorable",
        ])

        for line in fin:
            parsed = parse_pred_line(line)
            if parsed is None:
                continue

            left_id, right_id, pred = parsed
            total_pairs += 1
            if pred == "1":
                clone_ones += 1

            lm = func_map.get(left_id)
            rm = func_map.get(right_id)

            if lm:
                left_in, left_out, left_in_types, left_out_types, left_out_vars, left_ext = lm
            else:
                left_in = left_out = ""
                left_in_types = []
                left_out_types = []
                left_out_vars = []
                left_ext = None

            if rm:
                right_in, right_out, right_in_types, right_out_types, right_out_vars, right_ext = rm
            else:
                right_in = right_out = ""
                right_in_types = []
                right_out_types = []
                right_out_vars = []
                right_ext = None

            # Pair refactorable (both extractable)
            if left_ext is None or right_ext is None:
                pair_ref = ""
            else:
                pair_ref_bool = (left_ext is True) and (right_ext is True)
                pair_ref = "True" if pair_ref_bool else "False"
                if pair_ref_bool:
                    pair_refactorable_true += 1

            # Pair-level type compatibility
            if lm is None or rm is None:
                pair_in_ok = pair_out_ok = pair_type_ok = ""
            else:
                pair_type = compute_pair_type_compatibility(
                    {"InType": left_in_types, "OutType": left_out_types, "Out": left_out_vars},
                    {"InType": right_in_types, "OutType": right_out_types, "Out": right_out_vars},
                )
                pair_in_ok = "True" if pair_type["pair_inType_ok"] else "False"
                pair_out_ok = "True" if pair_type["pair_outType_ok"] else "False"
                pair_type_ok = "True" if pair_type["pair_type_ok"] else "False"
                if pair_type["pair_type_ok"]:
                    pair_type_ok_true += 1

            writer.writerow([
                left_id,
                right_id,
                pred,
                left_in,
                right_in,
                left_out,
                right_out,
                pair_in_ok,
                pair_out_ok,
                pair_type_ok,
                bool_to_str(left_ext),
                bool_to_str(right_ext),
                pair_ref,
            ])

    print("\n===== STATISTICS =====")
    print(f"Total parsed pairs          : {total_pairs}")
    print(f"clone_predict == 1          : {clone_ones}")
    print(f"pair_refactorable == True   : {pair_refactorable_true}")
    print(f"pair_type_ok == True        : {pair_type_ok_true}")

    if total_pairs > 0:
        print("\nRates:")
        print(f"  Clone positive rate       : {clone_ones / total_pairs:.6f}")
        print(f"  Refactorable rate         : {pair_refactorable_true / total_pairs:.6f}")
        print(f"  Pair type OK rate         : {pair_type_ok_true / total_pairs:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())