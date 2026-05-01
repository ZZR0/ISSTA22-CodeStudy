# AST_Clone_Extractability/main.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from AST_Clone_Extractability.io_nicad import load_nicad, write_output
from AST_Clone_Extractability.index_methods import FileMethodIndex
from AST_Clone_Extractability.rw_vars import extract_rw_by_region
from AST_Clone_Extractability.feasibility import compute_in_out, decide_extractable
from AST_Clone_Extractability.hazards import detect_cf_hazard_detail


REG_PRE = "CloneRegion_pre"
REG_WITHIN = "CloneRegion_within"
REG_POST = "CloneRegion_post"


def _parse_range(rng: str) -> Tuple[int, int]:
    a, b = rng.split("-")
    return int(a), int(b)


def _method_contains_range(method_info: Dict[str, Any], clone_start: int, clone_end: int) -> bool:
    s = method_info.get("start_line")
    e = method_info.get("end_line")
    if s is None or e is None:
        return False
    return s <= clone_start and e >= clone_end


def _resolve_enclosing_method(
    *,
    parser,
    methods: Dict[str, Any],
    clone_start: int,
    clone_end: int,
) -> Tuple[Any, Optional[str], Optional[Dict[str, Any]], bool]:
    """
    Find the *enclosing* method node that contains [clone_start, clone_end].
    Chooses the smallest enclosing method by span.
    Returns:
      (method_node, resolved_qname, resolved_method_info, resolved_ok)
    """
    best_mr = None
    best_q = None
    best_span = None

    for q, mr in methods.items():
        mi = mr.method_info
        if not _method_contains_range(mi, clone_start, clone_end):
            continue
        s = mi["start_line"]
        e = mi["end_line"]
        span = e - s
        if best_mr is None or span < best_span:
            best_mr = mr
            best_q = q
            best_span = span

    if best_mr is not None:
        return best_mr.node, best_q, best_mr.method_info, True

    # fallback
    return parser.root, None, None, False


def _region_ranges(fun_start: int, fun_end: int, clone_start: int, clone_end: int) -> Dict[str, str]:
    """
    Produce explicit ranges for pre/within/post in "a-b" form.
    If a region is empty, represent it as "" (empty string).
    """
    def mk(a: int, b: int) -> str:
        return f"{a}-{b}" if a <= b else ""

    return {
        REG_PRE: mk(fun_start, clone_start - 1),
        REG_WITHIN: mk(clone_start, clone_end),
        REG_POST: mk(clone_end + 1, fun_end),
    }


def analyze_nicad(
    classes: List[Dict[str, Any]],
    P: int,
    R: int,
    debug_hazard: bool = False,
) -> List[Dict[str, Any]]:
    idx = FileMethodIndex()
    out: List[Dict[str, Any]] = []

    for cls in classes:
        classid = cls.get("classid")
        nclones = cls.get("nclones")
        sim = cls.get("similarity")
        sources = cls.get("sources", [])

        new_sources: List[Dict[str, Any]] = []
        for inst in sources:
            file_path = inst["file"]
            rng = inst["range"]
            clone_start, clone_end = _parse_range(rng)

            parser, methods = idx.get(file_path)

            # Always resolve the enclosing function by clone range containment
            method_node, resolved_qname, resolved_mi, resolved_ok = _resolve_enclosing_method(
                parser=parser,
                methods=methods,
                clone_start=clone_start,
                clone_end=clone_end,
            )

            # Determine enclosing function start/end lines for region partitioning
            if resolved_ok and resolved_mi is not None:
                fun_start = int(resolved_mi["start_line"])
                fun_end = int(resolved_mi["end_line"])
            else:
                # If we can't resolve a function, we can't define meaningful pre/post.
                # Fallback to clone range only.
                fun_start, fun_end = clone_start, clone_end

            # RW extraction uses the enclosing method_node (so pre/post are within that method scope)
            rw = extract_rw_by_region(parser, method_node, clone_start, clone_end, only_method_scope=True)

            In, Out = compute_in_out(rw)
            cf_hazard, hazard_detail = detect_cf_hazard_detail(method_node, clone_start, clone_end)
            extractable = decide_extractable(In, Out, cf_hazard, P=P, R=R)

            if debug_hazard and cf_hazard and hazard_detail:
                print("\n" + "=" * 80)
                print("⚠ CONTROL-FLOW HAZARD DETECTED")
                print("-" * 80)
                print(f"Class ID        : {classid}")
                print(f"PCID            : {inst.get('pcid')}")
                print(f"File            : {file_path}")
                print(f"Clone Range     : {rng}")
                print(f"Enclosing Method: {resolved_qname or 'N/A'} (ok={resolved_ok})")
                print(f"Hazard Type     : {hazard_detail.get('type')}")
                print(f"Hazard Line     : {hazard_detail.get('line')}")
                print(f"Extractable     : {extractable}")
                print("=" * 80)

            # IMPORTANT: keep original inst["function"] exactly as input
            new_inst = dict(inst)

            # Add enclosing function info (the one used for analysis / pre-post)
            new_inst["enclosing_function"] = {
                "qualified_name": resolved_qname,
                "fun_range": f"{fun_start}-{fun_end}",
                "fun_nlines": (fun_end - fun_start + 1) if fun_end >= fun_start else 0,
                "resolved_ok": resolved_ok,
            }

            # Add explicit region ranges derived from enclosing function range + clone range
            new_inst["region_ranges"] = _region_ranges(fun_start, fun_end, clone_start, clone_end)

            # locals in this method (declared locals + parameters)
            new_inst["locals_in_method"] = sorted(rw.locals_in_method | rw.params_in_method)

            # V_r / V_w per region
            new_inst[REG_PRE] = {
                "var_read": sorted(rw.vr[REG_PRE]),
                "var_write": sorted(rw.vw[REG_PRE]),
            }
            new_inst[REG_WITHIN] = {
                "var_read": sorted(rw.vr[REG_WITHIN]),
                "var_write": sorted(rw.vw[REG_WITHIN]),
            }
            new_inst[REG_POST] = {
                "var_read": sorted(rw.vr[REG_POST]),
                "var_write": sorted(rw.vw[REG_POST]),
            }

            # feasibility
            new_inst["In"] = sorted(In)
            new_inst["Out"] = sorted(Out)
            new_inst["CFHazard"] = cf_hazard
            new_inst["CFHazard_detail"] = hazard_detail
            new_inst["Extractable"] = extractable
            new_inst["P"] = P
            new_inst["R"] = R

            new_sources.append(new_inst)

        out.append(
            {
                "classid": classid,
                "nclones": nclones,
                "similarity": sim,
                "sources": new_sources,
            }
        )

    return out


def print_extractable_stats(result: List[Dict[str, Any]]) -> None:
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

    def pct(x: int) -> float:
        return (100.0 * x / total) if total else 0.0

    print("\n=== Extractable Statistics ===")
    print(f"Total clone instances : {total}")
    print(f"Extractable = True    : {true_count} ({pct(true_count):.1f}%)")
    print(f"Extractable = False   : {false_count} ({pct(false_count):.1f}%)")
    print("================================\n")


def main() -> None:
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

    output_data = result if len(result) != 1 else result[0]
    output_path = Path(args.output).resolve()
    write_output(output_path, output_data, jsonl=args.jsonl)

    print(f"\nOutput written to: {args.output}\n")


if __name__ == "__main__":
    main()