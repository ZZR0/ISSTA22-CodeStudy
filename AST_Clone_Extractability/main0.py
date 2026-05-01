# AST_Clone_Extractability/main.py
from __future__ import annotations
import argparse
from typing import Any, Dict, List
from pathlib import Path

from AST_Clone_Extractability.io_nicad import load_nicad, write_output
from AST_Clone_Extractability.index_methods import FileMethodIndex
from AST_Clone_Extractability.rw_vars import extract_rw_by_region
from AST_Clone_Extractability.hazards import detect_cf_hazard
from AST_Clone_Extractability.feasibility import compute_in_out, decide_extractable
from AST_Clone_Extractability.hazards import detect_cf_hazard_detail


def _parse_range(rng: str) -> (int, int):
    a, b = rng.split("-")
    return int(a), int(b)


def analyze_nicad(classes: List[Dict[str, Any]], P: int, R: int, debug_hazard: bool = False) -> List[Dict[str, Any]]:
    idx = FileMethodIndex()
    out: List[Dict[str, Any]] = []

    # print("\n" + "#" * 80)
    # print("# CLONE FEASIBILITY DEBUG REPORT")
    # print("#" * 80)

    for cls in classes:
        classid = cls.get("classid")
        nclones = cls.get("nclones")
        sim = cls.get("similarity")
        sources = cls.get("sources", [])

        new_sources = []
        for inst in sources:
            file_path = inst["file"]
            rng = inst["range"]
            qname = inst.get("qualified_name")

            clone_start, clone_end = _parse_range(rng)

            parser, methods = idx.get(file_path)

            # Prefer qualified_name to locate method node; fallback by scanning line containment if missing.
            method_node = None
            if qname and qname in methods:
                method_node = methods[qname].node
            else:
                # Fallback: find any method whose [start_line, end_line] covers clone range
                for mr in methods.values():
                    s = mr.method_info["start_line"]
                    e = mr.method_info["end_line"]
                    if s <= clone_start and e >= clone_end:
                        method_node = mr.node
                        break

            # If still None, analyze at file root (rare)
            if method_node is None:
                method_node = parser.root

            rw = extract_rw_by_region(parser, method_node, clone_start, clone_end, only_method_scope=True)
            In, Out = compute_in_out(rw)
            cf_hazard, hazard_detail = detect_cf_hazard_detail(method_node, clone_start, clone_end)
            extractable = decide_extractable(In, Out, cf_hazard, P=P, R=R)

            # DEBUG PRINT HERE
            if debug_hazard and cf_hazard and hazard_detail:
                print("\n" + "=" * 80)
                print("⚠ CONTROL-FLOW HAZARD DETECTED")
                print("-" * 80)
                print(f"Class ID      : {classid}")
                print(f"PCID          : {inst.get('pcid')}")
                print(f"File          : {file_path}")
                print(f"Clone Range   : {rng}")
                print(f"Method        : {qname or 'N/A'}")
                print(f"Hazard Type   : {hazard_detail['type']}")
                print(f"Hazard Line   : {hazard_detail['line']}")
                print(f"Extractable   : {extractable}")
                print("=" * 80)

            new_inst = dict(inst)
            # locals in this method (declared locals + parameters)
            new_inst["locals_in_method"] = sorted(rw.locals_in_method | rw.params_in_method)

            # V_r / V_w per region (your required outputs)
            new_inst["CloneRegion_pre"] = {
                "var_read": sorted(rw.vr["CloneRegion_pre"]),
                "var_write": sorted(rw.vw["CloneRegion_pre"]),
            }
            new_inst["CloneRegion_within"] = {
                "var_read": sorted(rw.vr["CloneRegion_within"]),
                "var_write": sorted(rw.vw["CloneRegion_within"]),
            }
            new_inst["CloneRegion_post"] = {
                "var_read": sorted(rw.vr["CloneRegion_post"]),
                "var_write": sorted(rw.vw["CloneRegion_post"]),
            }

            # Paper-derived feasibility artifacts
            new_inst["In"] = sorted(In)
            new_inst["Out"] = sorted(Out)
            new_inst["CFHazard"] = cf_hazard
            new_inst["Extractable"] = extractable
            new_inst["P"] = P
            new_inst["R"] = R
            new_inst["CFHazard_detail"] = hazard_detail
            new_sources.append(new_inst)

        out.append({
            "classid": classid,
            "nclones": nclones,
            "similarity": sim,
            "sources": new_sources
        })

    return out

def print_extractable_stats(result):
    total = 0
    true_count = 0
    false_count = 0

    for cls in result:
        for inst in cls.get("sources", []):
            total += 1
            if inst.get("Extractable", False):
                true_count += 1
            else:
                false_count += 1

    pct = lambda x: (100.0 * x / total) if total else 0.0

    print("\n=== Extractable Statistics ===")
    print(f"Total clone instances : {total}")
    print(f"Extractable = True    : {true_count} ({pct(true_count):.1f}%)")
    print(f"Extractable = False   : {false_count} ({pct(false_count):.1f}%)")
    print("================================\n")

def main():
    ap = argparse.ArgumentParser(description="NiCad -> AST feasibility analysis (In/Out/CFHazard/Extractable)")
    ap.add_argument("--input", required=True, help="NiCad JSON or JSONL")
    ap.add_argument("--output", required=True, help="Output JSON/JSONL")
    ap.add_argument("--jsonl", action="store_true", help="Write JSONL output")
    ap.add_argument("--P", type=int, default=7, help="Max allowed parameter count (|In(i)|)")
    ap.add_argument("--R", type=int, default=1, help="Max allowed return count (|Out(i)|). Java default is 1.")
    ap.add_argument("--debug-hazard", action="store_true", help="Print hazard details per clone instance")
    args = ap.parse_args()

    classes = load_nicad(args.input)
    result = analyze_nicad(classes, P=args.P, R=args.R, debug_hazard=args.debug_hazard)
    print_extractable_stats(result if isinstance(result, list) else [result])

    # If input was a single object, you may prefer output single object. Here we keep list for consistency.
    output_data = result if len(result) != 1 else result[0]
    output_path = Path(args.output).resolve()
    write_output(output_path, output_data, jsonl=args.jsonl)

    print(f"\nOutput written to: {args.output}\n")



if __name__ == "__main__":
    main()
