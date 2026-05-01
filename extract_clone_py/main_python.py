import re
import argparse
import json
import sys
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import keyword

from python_treesitter_parser import PythonTreeSitterParser
from util_ast_python import (
    extract_rw_by_region, REG_PRE, REG_WITHIN, REG_POST,
    iter_descendants, method_name, collect_params, collect_local_vars, return_type
)

def detect_cf_hazard_detail(enclosing_node, clone_start: int, clone_end: int) -> Tuple[bool, str]:
    for node in iter_descendants(enclosing_node):
        if node.type in ["break_statement", "continue_statement"]:
            # Only flag if it's actually detached from a loop
            curr = node
            in_loop = False
            while curr and curr != enclosing_node:
                if curr.type in ["for_statement", "while_statement"]:
                    in_loop = True
                    break
                curr = curr.parent
            if not in_loop:
                return True, "Syntax Error: break/continue outside loop"
    return False, ""

@dataclass
class MethodRecord:
    node: Any
    method_info: Dict[str, Any]

class FileMethodIndexPython:
    def get(self, file_path: str) -> Tuple[PythonTreeSitterParser, List[MethodRecord]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            src = f.read()
        parser = PythonTreeSitterParser(src)
        methods = []  # Changed from dict to list to prevent name collisions
        
        for node in iter_descendants(parser.root):
            if node.type == "function_definition":
                m_name = method_name(parser, node)
                m_start = node.start_point[0] + 1
                m_end = node.end_point[0] + 1
                params = collect_params(parser, node)
                locals_list = collect_local_vars(parser, node)
                r_type = return_type(parser, node) or "None"
                
                record = MethodRecord(
                    node=node,
                    method_info={
                        "qualified": m_name, "start_line": m_start, "end_line": m_end,
                        "parameters": params, "local_variables": locals_list, "return_type": r_type
                    }
                )
                methods.append(record)  # Append to list instead of overwriting dictionary keys
        return parser, methods
# ============================================================================
def compute_extractable(
    *,
    clone_node,
    cf_hazard: bool,
    is_full: bool,
    require_non_full: bool = False,
) -> bool:
    # Must have a located AST node for the clone
    if clone_node is None:
        return False
    # Hard control-flow hazard (break/continue outside loop etc.)
    if cf_hazard:
        return False
    # Optional policy: exclude full-function clones
    if require_non_full and is_full:
        return False
    return True 

def is_full_function_robust(enclosing_node, clone_start, clone_end, lead_slack=1, tail_slack=5):
    func_start = enclosing_node.start_point[0] + 1
    func_end = enclosing_node.end_point[0] + 1
    return (clone_start <= func_start + lead_slack) and (clone_end >= func_end - 1)

def find_enclosing_method(methods, clone_start: int, clone_end: int):
    best_record = None
    max_overlap = 0
    
    # Iterate directly over the list of MethodRecords
    for record in methods:
        m_info = record.method_info
        
        # Ensure we have valid integers for start/end
        try:
            m_start = int(m_info.get("start_line", 0))
            m_end = int(m_info.get("end_line", 0))
        except (ValueError, TypeError):
            continue
        
        if m_start == 0 or m_end == 0: 
            continue
            
        # Calculate the intersection (overlap) between the method and the clone
        overlap_start = max(m_start, clone_start)
        overlap_end = min(m_end, clone_end)
        
        # Calculate number of overlapping lines
        current_overlap = max(0, overlap_end - overlap_start + 1)
        
        # Keep the method with the largest overlap
        if current_overlap > max_overlap:
            max_overlap = current_overlap
            best_record = record
            
    return best_record

def _param_base_name(param_str: str) -> str:
    s = str(param_str).strip()
    s = s.split(":", 1)[0].strip()     # drop annotation
    s = s.split("=", 1)[0].strip()     # drop default
    s = s.lstrip("*")                  # *args, **kwargs -> args, kwargs
    return s

def clone_has_return(clone_node) -> bool:
    if not clone_node:
        return False
    for n in iter_descendants(clone_node):
        if n.type == "return_statement":
            return True
    return False

def find_node_by_range(root_node, start_line, end_line):
    target = None
    def _search(node):
        nonlocal target
        n_start = node.start_point[0] + 1
        n_end = node.end_point[0] + 1
        if n_end < start_line or n_start > end_line: return
        if n_start <= start_line and n_end >= end_line:
            if target is None or (node.end_byte - node.start_byte) < (target.end_byte - target.start_byte):
                target = node
                for child in node.children: _search(child)
    _search(root_node)
    return target

def has_return_in_line_range(enclosing_node, start_line: int, end_line: int) -> bool:
    """Return True if any return_statement appears within [start_line, end_line]."""
    for n in iter_descendants(enclosing_node):
        if n.type == "return_statement":
            ln = n.start_point[0] + 1
            if start_line <= ln <= end_line:
                return True
    return False

def load_template(template_name: str) -> str:
    path = Path("templates") / template_name
    if not path.is_file():
        print(f"Error: Required template file '{path}' is missing.")
        sys.exit(1)
    return path.read_text(encoding="utf-8")

def fmt_set(var_set):
    return html.escape(", ".join(sorted(var_set))) if var_set else "<span style='color:#aaa'>None</span>"

def fmt_math_set_for_html(s):
    return "{" + html.escape(", ".join(sorted(s))) + "}" if s else "{}"

def _split_var_and_line(s: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^(.*?)\s*\(Line\s+(\d+)\)\s*$", s.strip())
    if not m: return s.strip(), None
    return m.group(1).strip(), int(m.group(2))

def _base_names_from_with_line(items) -> List[str]:
    bases = set()
    for it in items or []:
        b, _ = _split_var_and_line(str(it))
        if b: bases.add(b)
    return sorted(bases)

def _sorted_list(items) -> List[str]:
    parsed = []
    for it in items or []:
        s = str(it)
        b, ln = _split_var_and_line(s)
        parsed.append((b, ln if ln is not None else 10**9, s))
    parsed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[2] for t in parsed]

def _build_type_map(m_info: Dict[str, Any]) -> Dict[str, str]:
    type_map: Dict[str, str] = {}
    for param_str in m_info.get("parameters", []) or []:
        name = _param_base_name(param_str)
        if name:
            type_map[name] = "Any"
    for var_type, var_name, _ in m_info.get("local_variables", []) or []:
        type_map[str(var_name)] = str(var_type)
    return type_map

def _derive_def_before(m_info: Dict[str, Any], clone_start: int) -> List[str]:
    s = set()
    for param_str in m_info.get("parameters", []) or []:
        name = _param_base_name(param_str)
        if name:
            s.add(name)
        
    for _, var_name, line_num in m_info.get("local_variables", []) or []:
        try: ln = int(line_num)
        except Exception: continue
        if ln < clone_start: s.add(str(var_name))
    return sorted(s)

def _derive_def_within(rw_regions) -> List[str]: return _base_names_from_with_line(rw_regions.vw.get(REG_WITHIN, set()))
def _derive_use_after(rw_regions) -> List[str]: return _base_names_from_with_line(rw_regions.vr.get(REG_POST, set()))
def _derive_use_within(rw_regions) -> List[str]: return _base_names_from_with_line(rw_regions.vr.get(REG_WITHIN, set()))

def _infer_signature_python(in_vars: List[str], out_vars: List[str], type_map: Dict[str, str], original_return: str = "None", has_return_stmt: bool = False) -> Tuple[str, str, List[str], List[str]]:
    in_types = [type_map.get(v, "Any") for v in in_vars]

    if "__return__" in out_vars:
        return_type_str = original_return if original_return and original_return != "None" else "Any"
        out_types = [return_type_str]

    elif len(out_vars) == 0:
        return_type_str = "None"
        out_types = []

    elif len(out_vars) == 1:
        t = type_map.get(out_vars[0], "Any")
        return_type_str = t
        out_types = [t]

    else:
        out_types = [type_map.get(v, "Any") for v in out_vars]
        return_type_str = "Tuple[" + ", ".join(out_types) + "]"
    
    extracted_param_list = [f"{v}: {type_map.get(v, 'Any')}" for v in in_vars]
    params_str = ", ".join(extracted_param_list)
    extracted_signature_str = f"def extracted_clone({params_str}) -> {return_type_str}:"
    return return_type_str, extracted_signature_str, in_types, out_types

def process_clone_jsonl(jsonl_path: str, base_dir: str, output_html: str, output_jsonl: str):
    """
    Fully improved version:
    - Fixes param/default issues via _derive_def_before() (you already updated it).
    - Uses AST-based return detection (clone_has_return) and a stable RETURN_MARKER.
    - Builds out_set BEFORE any hazard checks and avoids df_hazard by default.
    - Uses actual clone text (from parsed method_source) for any optional return-var heuristics.
    - Adds a few safety guards (range slicing, template formatting).
    """
    jsonl_file, base_path = Path(jsonl_path), Path(base_dir)
    if not jsonl_file.is_file():
        sys.exit(1)

    tmpl_header = load_template("header.html")
    tmpl_footer = load_template("footer.html")
    tmpl_class_start = load_template("class_start.html")
    tmpl_instance_meta = load_template("instance_meta.html")
    tmpl_instance_error = load_template("instance_error.html")

    indexer = FileMethodIndexPython()
    html_content = [tmpl_header]

    out_jsonl_path = Path(output_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    written_classes, written_sources, total_classes_processed = 0, 0, 0
    dropped_full_functions, dropped_hazards, dropped_type_mismatches, dropped_classes = 0, 0, 0, 0

    RETURN_MARKER = "__return__"

    with open(jsonl_file, "r", encoding="utf-8") as f_in, open(out_jsonl_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                clone_class = json.loads(line)
            except json.JSONDecodeError:
                continue

            class_id = clone_class.get("classid")
            sources = clone_class.get("sources", [])
            total_classes_processed += 1

            augmented_class = dict(clone_class)
            augmented_sources = []
            class_html_buffer = []
            blueprint_in_types, blueprint_out_types = None, None
            blueprint_in_vars, blueprint_out_vars = None, None

            for source in sources:
                src_aug = dict(source)
                rel_file = source.get("file")
                range_str = source.get("range")
                func_id = source.get("func_id")

                if not rel_file or not range_str:
                    continue

                try:
                    c_start_str, c_end_str = str(range_str).split("-")
                    clone_start, clone_end = int(c_start_str), int(c_end_str)
                except ValueError:
                    continue

                full_file_path = base_path / rel_file
                if not full_file_path.is_file():
                    class_html_buffer.append(
                        tmpl_instance_error.format(
                            func_id=func_id,
                            rel_file=rel_file,
                            range_str=range_str,
                            error_msg="File not found on disk."
                        )
                    )
                    continue

                try:
                    parser, methods = indexer.get(str(full_file_path))
                    enclosing_record = find_enclosing_method(methods, clone_start, clone_end)

                    if not enclosing_record:
                        class_html_buffer.append(
                            tmpl_instance_error.format(
                                func_id=func_id,
                                rel_file=rel_file,
                                range_str=range_str,
                                error_msg="No enclosing method found bounds."
                            )
                        )
                        continue

                    enclosing_node = enclosing_record.node
                    m_info = enclosing_record.method_info

                    m_start = int(m_info.get("start_line"))
                    m_end = int(m_info.get("end_line"))

                    func_code = parser.text_of(enclosing_record.node)
                    enclosing_info = {
                        "qualified_name": m_info.get("qualified"),
                        "fun_range": f"{m_start}-{m_end}",
                        "fun_nlines": (m_end - m_start) + 1,
                        "func_code": func_code,
                    }

                    clone_node = find_node_by_range(enclosing_node, clone_start, clone_end)
                    is_full = is_full_function_robust(enclosing_node, clone_start, clone_end)
                    if is_full: dropped_full_functions += 1; continue

                    rw_regions = extract_rw_by_region(
                        parser, enclosing_record.node,
                        clone_start, clone_end,
                        only_method_scope=True
                    )

                    type_map = _build_type_map(m_info)
                    type_map.update(getattr(rw_regions, "field_types", {}) or {})

                    # ===== IN =====
                    use_set = _derive_use_within(rw_regions)
                    def_before_set = set(_derive_def_before(m_info, clone_start))
                    def_before_set.update(getattr(rw_regions, "fields_in_class", set()) or set())
                    in_set = sorted(set(use_set).intersection(def_before_set))

                    # ===== OUT (dataflow + return marker) =====
                    def_within_set = _derive_def_within(rw_regions)
                    use_after_set = _derive_use_after(rw_regions)

                    # Robust return detection: scan return statements by line range
                    has_return_stmt = has_return_in_line_range(enclosing_node, clone_start, clone_end)

                    # If this is marked "full function", also accept return anywhere in the function body
                    # (helps if clone range slightly misses a return line)
                    if is_full and not has_return_stmt:
                        has_return_stmt = has_return_in_line_range(enclosing_node, m_start, m_end)

                    # Base OUT: defs in clone used after clone
                    out_set = set(def_within_set).intersection(use_after_set)

                    # Add return marker if clone has any return statement
                    if has_return_stmt:
                        out_set.add(RETURN_MARKER)

                    out_set = sorted(out_set)
                    # ===== Enforce consistent var lists across clone instances in this class =====
                    if blueprint_in_vars is None:
                        blueprint_in_vars = list(in_set)
                        blueprint_out_vars = list(out_set)
                    else:
                        # Force current instance to match the blueprint signature
                        in_set = list(blueprint_in_vars)
                        out_set = list(blueprint_out_vars)

                    # ===== Hazards =====
                    cf_hazard, cf_detail = detect_cf_hazard_detail(enclosing_record.node, clone_start, clone_end)

                    # Default: DO NOT drop on multiple outputs (df_hazard)
                    # If you ever want df_hazard, use:
                    # df_hazard = len([x for x in out_set if x != RETURN_MARKER]) > 1
                    # is_hazardous = cf_hazard or df_hazard
                    is_hazardous = cf_hazard
                    extractable = (clone_node is not None) and (not cf_hazard) and (not is_full)

                    if is_hazardous:
                        dropped_hazards += 1
                        continue

                    # ===== Signature inference =====
                    original_return_type = m_info.get("return_type", "None")
                    return_type_str, extracted_sig, in_types, out_types = _infer_signature_python(
                        in_set, out_set, type_map, original_return_type, has_return_stmt
                    )

                    # ===== Cross-instance consistency check =====
                    if blueprint_in_vars is not None and (list(in_set) != blueprint_in_vars or list(out_set) != blueprint_out_vars):
                        print(f"[DEBUG] Signature normalized in class {class_id} (func_id: {func_id}) "
                            f"IN {in_set}->{blueprint_in_vars}, OUT {out_set}->{blueprint_out_vars}")

                    # ===== Persist augmented fields =====
                    src_aug.update({
                        "is_full_function_clone": is_full,
                        "cf_hazard": cf_hazard,
                        "extractable": extractable,
                        "Extracted Signature": extracted_sig,
                        "ReturnType": return_type_str,
                        "In": {
                            "In(i)": in_set,
                            "InType": {
                                "InType": in_types,
                                "Use(i)": use_set,
                                "Defbefore(i)": sorted(def_before_set),
                            },
                        },
                        "Out": {
                            "Out(i)": out_set,
                            "OutType": out_types,
                            "Defwithin(i)": def_within_set,
                            "Useafter(i)": use_after_set,
                        },
                        "enclosing_function": enclosing_info,
                    })

                    # ===== HTML emission =====
                    extracted_params_str = html.escape(", ".join([f"{v}: {type_map.get(v, 'Any')}" for v in in_set]))

                    class_html_buffer.append(
                        tmpl_instance_meta.format(
                            func_id=func_id,
                            rel_file=rel_file,
                            range_str=range_str,
                            method_qualified=m_info.get("qualified"),
                            m_start=m_start,
                            m_end=m_end,
                            return_type=html.escape(return_type_str),
                            extracted_params=extracted_params_str,
                            extracted_sig=html.escape(extracted_sig),
                            use_set_str=fmt_math_set_for_html(set(use_set)),
                            def_set_str=fmt_math_set_for_html(set(def_before_set)),
                            in_set_str=fmt_math_set_for_html(set(in_set)),
                            def_within_str=fmt_math_set_for_html(set(def_within_set)),
                            use_after_str=fmt_math_set_for_html(set(use_after_set)),
                            out_set_str=fmt_math_set_for_html(set(out_set)),
                            vr_pre=fmt_set(set(_sorted_list(rw_regions.vr.get(REG_PRE, set())))),
                            vr_within=fmt_set(set(_sorted_list(rw_regions.vr.get(REG_WITHIN, set())))),
                            vr_post=fmt_set(set(_sorted_list(rw_regions.vr.get(REG_POST, set())))),
                            vw_pre=fmt_set(set(_sorted_list(rw_regions.vw.get(REG_PRE, set())))),
                            vw_within=fmt_set(set(_sorted_list(rw_regions.vw.get(REG_WITHIN, set())))),
                            vw_post=fmt_set(set(_sorted_list(rw_regions.vw.get(REG_POST, set())))),
                        )
                    )

                    method_source = parser.text_of(enclosing_record.node)
                    source_lines = method_source.split("\n")
                    in_signature = True
                    use_set_for_highlight = set(use_set)

                    for i, raw_line in enumerate(source_lines):
                        current_line_num = m_start + i
                        escaped_line = html.escape(raw_line)

                        if clone_start <= current_line_num <= clone_end:
                            line_class = "in-clone"
                            in_signature = False
                            for var in sorted(use_set_for_highlight, key=len, reverse=True):
                                escaped_line = re.sub(
                                    rf"\b({re.escape(var)})\b",
                                    r'<span style="background-color: #ffeb3b; color: #b30000; border-radius: 2px; padding: 0 2px;">\1</span>',
                                    escaped_line,
                                )
                        elif current_line_num < clone_start:
                            if in_signature:
                                line_class = "method-signature"
                                if ":" in raw_line:
                                    in_signature = False
                            else:
                                line_class = "before-clone"
                        else:
                            line_class = "after-clone"

                        class_html_buffer.append(
                            f'<span class="line-number">{current_line_num}</span><span class="{line_class}">{escaped_line}</span>\n'
                        )

                    class_html_buffer.append("</code></pre>\n</div>\n")

                    augmented_sources.append(src_aug)
                    written_sources += 1

                except Exception as e:
                    print(f"  > Error processing {func_id}: {e}")

            if len(augmented_sources) < 2: dropped_classes += 1; continue

            augmented_class["sources"] = augmented_sources
            augmented_class["nclones"] = len(augmented_sources)

            html_content.append(tmpl_class_start.format(class_id=class_id, instance_count=len(augmented_sources)))
            html_content.extend(class_html_buffer)
            html_content.append("</div>\n")

            f_out.write(json.dumps(augmented_class, ensure_ascii=False) + "\n")
            written_classes += 1

    html_content.append(tmpl_footer)
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(html_content), encoding="utf-8")

    print("\n--- Filtering & Generation Summary (PYTHON) ---")
    print(f"Total classes processed:             {total_classes_processed}")
    print(f"--------------------------------------")
    print(f"Classes written successfully:        {written_classes}")
    print(f"Sources analyzed successfully:       {written_sources}")
    print(f"\nSuccessfully wrote HTML visualization to: {output_path.absolute()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="data/clone_visualization.html")
    parser.add_argument("--out-jsonl", dest="out_jsonl", type=str, default=None)

    args = parser.parse_args()
    out_jsonl = args.out_jsonl or str(Path(args.output).with_suffix(".jsonl"))
    process_clone_jsonl(args.jsonl, args.base_dir, args.output, out_jsonl)

if __name__ == "__main__":
    main()