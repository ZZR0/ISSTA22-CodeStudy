import re
import argparse
import json
import sys
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from index_methods import FileMethodIndex
from util_ast import extract_rw_by_region, REG_PRE, REG_WITHIN, REG_POST
from hazards import detect_cf_hazard_detail

def find_enclosing_method(methods, clone_start: int, clone_end: int):
    for qualified_name, record in methods.items():
        m_info = record.method_info
        m_start = m_info.get("start_line")
        m_end = m_info.get("end_line")

        if m_start and m_end:
            if m_start <= clone_start and m_end >= clone_end:
                return record
    return None

def find_node_by_range(root_node, start_line, end_line):
    """
    Finds the smallest AST node that completely contains the target line range.
    Optimized to skip branches that are completely out of bounds.
    """
    target = None
    
    def _search(node):
        nonlocal target
        n_start = node.start_point[0] + 1
        n_end = node.end_point[0] + 1
        
        # Optimization: If this node is completely outside our range, stop searching its children
        if n_end < start_line or n_start > end_line:
            return

        # Check if this node encompasses our clone range
        if n_start <= start_line and n_end >= end_line:
            if target is None or (node.end_byte - node.start_byte) < (target.end_byte - target.start_byte):
                target = node
                for child in node.children:
                    _search(child)
            
    _search(root_node)
    return target

def _range_or_none(start: int, end: int) -> Optional[str]:
    return f"{start}-{end}" if start <= end else None

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
    if not m:
        return s.strip(), None
    return m.group(1).strip(), int(m.group(2))

def _base_names_from_with_line(items) -> List[str]:
    bases = set()
    for it in items or []:
        b, _ = _split_var_and_line(str(it))
        if b:
            bases.add(b)
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
        parts = str(param_str).strip().rsplit(" ", 1)
        if len(parts) == 2:
            type_map[parts[1]] = parts[0]
    for var_type, var_name, _ in m_info.get("local_variables", []) or []:
        type_map[str(var_name)] = str(var_type)
    return type_map

def _derive_def_before(m_info: Dict[str, Any], clone_start: int) -> List[str]:
    s = set()
    for param_str in m_info.get("parameters", []) or []:
        parts = str(param_str).strip().rsplit(" ", 1)
        if len(parts) == 2:
            s.add(parts[1])
        elif len(parts) == 1:
            s.add(parts[0])

    for var_type, var_name, line_num in m_info.get("local_variables", []) or []:
        try:
            ln = int(line_num)
        except Exception:
            continue
        if ln < clone_start:
            s.add(str(var_name))
    return sorted(s)

def _derive_def_within(rw_regions) -> List[str]:
    return _base_names_from_with_line(rw_regions.vw.get(REG_WITHIN, set()))

def _derive_use_after(rw_regions) -> List[str]:
    return _base_names_from_with_line(rw_regions.vr.get(REG_POST, set()))

def _derive_use_within(rw_regions) -> List[str]:
    return _base_names_from_with_line(rw_regions.vr.get(REG_WITHIN, set()))

def _infer_signature(in_vars: List[str], out_vars: List[str], type_map: Dict[str, str], original_return: str = "void", has_return_stmt: bool = False) -> Tuple[str, str, List[str], List[str]]:
    in_types = [type_map.get(v, "Object") for v in in_vars]

    if len(out_vars) == 0:
        if has_return_stmt and original_return and original_return != "void":
            return_type_str = original_return
        else:
            return_type_str = "void"
        out_types: List[str] = []
    elif len(out_vars) == 1:
        t = type_map.get(out_vars[0], "Object")
        return_type_str = t
        out_types = [t]
    else:
        out_types = [type_map.get(v, "Object") for v in out_vars]
        return_type_str = "Tuple<" + ", ".join(out_types) + ">"

    extracted_param_list = [f"{type_map.get(v, 'Object')} {v}" for v in in_vars]
    params_str = ", ".join(extracted_param_list)

    extracted_signature_str = f"public {return_type_str} extractedClone({params_str})"
    return return_type_str, extracted_signature_str, in_types, out_types


def process_clone_jsonl(jsonl_path: str, base_dir: str, output_html: str, output_jsonl: str):
    jsonl_file = Path(jsonl_path)
    base_path = Path(base_dir)

    if not jsonl_file.is_file():
        print(f"Error: The file '{jsonl_path}' does not exist.")
        sys.exit(1)

    tmpl_header = load_template("header.html")
    tmpl_footer = load_template("footer.html")
    tmpl_class_start = load_template("class_start.html")
    tmpl_instance_meta = load_template("instance_meta.html")
    tmpl_instance_error = load_template("instance_error.html")

    indexer = FileMethodIndex()
    html_content = [tmpl_header]

    out_jsonl_path = Path(output_jsonl)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    written_classes = 0
    written_sources = 0
    
    # Tracking for dropped items
    total_classes_processed = 0  # <--- NEW TRACKER
    dropped_full_functions = 0
    dropped_hazards = 0
    dropped_type_mismatches = 0  # <--- NEW TRACKER
    dropped_classes = 0

    with open(jsonl_file, "r", encoding="utf-8") as f_in, open(out_jsonl_path, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                clone_class = json.loads(line)
            except json.JSONDecodeError:
                continue

            class_id = clone_class.get("classid")
            sources = clone_class.get("sources", []) or []

            total_classes_processed += 1  # <--- INCREMENT HERE

            print(f"--- Processing Clone Class ID: {class_id} ---")

            augmented_class = dict(clone_class)
            augmented_sources: List[Dict[str, Any]] = []
            class_html_buffer = []

            # <--- NEW: Blueprint types for this specific clone class
            blueprint_in_types = None
            blueprint_out_types = None

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
                    class_html_buffer.append(tmpl_instance_error.format(func_id=func_id, rel_file=rel_file, range_str=range_str, error_msg="File not found on disk."))
                    continue

                try:
                    parser, methods = indexer.get(str(full_file_path))
                    enclosing_record = find_enclosing_method(methods, clone_start, clone_end)

                    if not enclosing_record:
                        class_html_buffer.append(tmpl_instance_error.format(func_id=func_id, rel_file=rel_file, range_str=range_str, error_msg="No enclosing method found bounds."))
                        continue

                    m_info = enclosing_record.method_info
                    m_start = int(m_info.get("start_line"))
                    m_end = int(m_info.get("end_line"))

                    # Extract the code block for the function using the parser
                    func_code = parser.text_of(enclosing_record.node)

                    # Construct the dictionary structure
                    enclosing_info = {
                        "qualified_name": m_info.get("qualified"),
                        "fun_range": f"{m_start}-{m_end}",
                        "fun_nlines": (m_end - m_start) + 1,
                        "func_code": func_code
                    }

                    # --- DETECT FULL FUNCTION CLONES ---
                    enclosing_node = enclosing_record.node
                    body_node = enclosing_node.child_by_field_name("body")
                    is_full = False

                    if body_node:
                        clone_node = find_node_by_range(enclosing_node, clone_start, clone_end)
                        if clone_node:
                            is_full = (clone_node.start_byte <= body_node.start_byte) and \
                                      (clone_node.end_byte >= body_node.end_byte)

                    # FILTERING: Drop Full Function Clones
                    if is_full:
                        print(f"  > Dropped {func_id}: Full function clone.")
                        dropped_full_functions += 1
                        continue

                    # --- Run Data-Flow Analysis FIRST ---
                    rw_regions = extract_rw_by_region(parser, enclosing_record.node, clone_start, clone_end, only_method_scope=True)

                    type_map = _build_type_map(m_info)
                    type_map.update(getattr(rw_regions, 'field_types', {}))

                    use_set = _derive_use_within(rw_regions)
                    
                    def_before_list = _derive_def_before(m_info, clone_start)
                    def_before_set = set(def_before_list)
                    def_before_set.update(getattr(rw_regions, 'fields_in_class', set()))

                    in_set = sorted(set(use_set).intersection(def_before_set))
                    def_within_set = _derive_def_within(rw_regions)
                    use_after_set = _derive_use_after(rw_regions)
                    
                    # We have our out_set!
                    out_set = sorted(set(def_within_set).intersection(use_after_set))

                    # --- Run Hazard Detection (Control Flow + Data Flow) ---
                    cf_hazard, cf_detail = detect_cf_hazard_detail(enclosing_record.node, clone_start, clone_end)
                    
                    # Check for Multiple Output hazard
                    df_hazard = len(out_set) > 1
                    df_detail = f"Multiple outputs ({', '.join(out_set)})" if df_hazard else ""

                    is_hazardous = cf_hazard or df_hazard

                    # FILTERING: Drop Hazardous Clones
                    if is_hazardous:
                        hazard_reasons = []
                        if cf_hazard: hazard_reasons.append(cf_detail)
                        if df_hazard: hazard_reasons.append(df_detail)
                        
                        hazard_str = " | ".join(hazard_reasons)
                        print(f"  > Dropped {func_id}: Extraction hazard detected -> {hazard_str}")
                        dropped_hazards += 1
                        continue

                    clone_code_str = src_aug.get("code", "")
                    has_return_stmt = bool(re.search(r'\breturn\b', clone_code_str))
                    original_return_type = m_info.get("return_type", "void")

                    return_type_str, extracted_sig, in_types, out_types = _infer_signature(
                        in_set, out_set, type_map, original_return_type, has_return_stmt
                    )

                    # <--- NEW: FILTERING: Drop Type Mismatches
                    if blueprint_in_types is None:
                        # Set the blueprint on the first successful, safe clone instance
                        blueprint_in_types = in_types
                        blueprint_out_types = out_types
                    else:
                        # Compare strict equality for subsequent instances
                        if in_types != blueprint_in_types or out_types != blueprint_out_types:
                            print(f"  > Dropped {func_id}: Type mismatch against blueprint (In: {in_types}, Out: {out_types})")
                            dropped_type_mismatches += 1
                            continue

                    method_source = parser.text_of(enclosing_record.node)
                    method_source = re.sub(r'^\s*@\w+(?:\([^)]*\))?\s*$', '', method_source, flags=re.MULTILINE)

                    # Append augmented data
                    src_aug.update({
                        "is_full_function_clone": is_full,
                        "cf_hazard": cf_hazard,
                        "Extracted Signature": extracted_sig,
                        "ReturnType": return_type_str,
                        "In": {"In(i)": in_set, "InType": {"InType": in_types, "Use(i)": use_set, "Defbefore(i)": sorted(def_before_set)}},
                        "Out": {"Out(i)": out_set, "OutType": out_types, "Defwithin(i)": def_within_set, "Useafter(i)": use_after_set},
                        "enclosing_function": enclosing_info
                    })

                    # --- HTML emission ---
                    extracted_params_str = html.escape(", ".join([f"{type_map.get(v, 'Object')} {v}" for v in in_set]))

                    class_html_buffer.append(
                        tmpl_instance_meta.format(
                            func_id=func_id, rel_file=rel_file, range_str=range_str,
                            method_qualified=m_info.get("qualified"), m_start=m_start, m_end=m_end,
                            return_type=html.escape(return_type_str), extracted_params=extracted_params_str,
                            use_set_str=fmt_math_set_for_html(set(use_set)), def_set_str=fmt_math_set_for_html(set(def_before_set)),
                            in_set_str=fmt_math_set_for_html(set(in_set)), def_within_str=fmt_math_set_for_html(set(def_within_set)),
                            use_after_str=fmt_math_set_for_html(set(use_after_set)), out_set_str=fmt_math_set_for_html(set(out_set)),
                            vr_pre=fmt_set(set(_sorted_list(rw_regions.vr.get(REG_PRE, set())))),
                            vr_within=fmt_set(set(_sorted_list(rw_regions.vr.get(REG_WITHIN, set())))),
                            vr_post=fmt_set(set(_sorted_list(rw_regions.vr.get(REG_POST, set())))),
                            vw_pre=fmt_set(set(_sorted_list(rw_regions.vw.get(REG_PRE, set())))),
                            vw_within=fmt_set(set(_sorted_list(rw_regions.vw.get(REG_WITHIN, set())))),
                            vw_post=fmt_set(set(_sorted_list(rw_regions.vw.get(REG_POST, set())))),
                        )
                    )

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
                                escaped_line = re.sub(rf"\b({re.escape(var)})\b", r'<span style="background-color: #ffeb3b; color: #b30000; border-radius: 2px; padding: 0 2px;">\1</span>', escaped_line)
                        elif current_line_num < clone_start:
                            if in_signature:
                                line_class = "method-signature"
                                if "{" in raw_line: in_signature = False
                            else: line_class = "before-clone"
                        else:
                            line_class = "after-clone"

                        class_html_buffer.append(f'<span class="line-number">{current_line_num}</span><span class="{line_class}">{escaped_line}</span>\n')

                    class_html_buffer.append("</code></pre>\n</div>\n")
                    augmented_sources.append(src_aug)
                    written_sources += 1

                except Exception as e:
                    print(f"  > Error processing {func_id}: {e}")

            # FILTERING: Drop Class if we filtered out too many instances
            if len(augmented_sources) < 2:
                print(f"  > Dropping Class {class_id}: Not enough valid clones left ({len(augmented_sources)}).")
                dropped_classes += 1
                continue

            # Update nclones to the newly filtered count
            augmented_class["sources"] = augmented_sources
            augmented_class["nclones"] = len(augmented_sources)

            # --- Commit HTML & JSONL ---
            html_content.append(tmpl_class_start.format(class_id=class_id, instance_count=len(augmented_sources)))
            html_content.extend(class_html_buffer)
            html_content.append("</div>\n")
            
            f_out.write(json.dumps(augmented_class, ensure_ascii=False) + "\n")
            written_classes += 1

    html_content.append(tmpl_footer)
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(html_content), encoding="utf-8")

    print("\n--- Filtering & Generation Summary ---")
    print(f"Total classes processed:             {total_classes_processed}") # <--- NEW
    print(f"--------------------------------------")
    print(f"Sources dropped (Full-function):     {dropped_full_functions}")
    print(f"Sources dropped (Hazards):           {dropped_hazards}")
    print(f"Sources dropped (Type Mismatch):     {dropped_type_mismatches}")
    print(f"Classes dropped (nclones < 2):       {dropped_classes}")
    print(f"--------------------------------------")
    print(f"Classes written successfully:        {written_classes}")
    print(f"Sources analyzed successfully:       {written_sources}")
    print(f"\nSuccessfully wrote visualization to: {output_path.absolute()}")
    print(f"Successfully wrote augmented JSONL to: {out_jsonl_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Find enclosing Java methods for clone blocks, visualize in HTML, and export augmented JSONL.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory to resolve the relative file paths.")
    parser.add_argument("--output", type=str, default="clone_visualization.html", help="Output HTML file name.")
    parser.add_argument("--out-jsonl", dest="out_jsonl", type=str, default=None, help="Output augmented JSONL file name.")

    args = parser.parse_args()
    out_jsonl = args.out_jsonl or str(Path(args.output).with_suffix(".jsonl"))
    process_clone_jsonl(args.jsonl, args.base_dir, args.output, out_jsonl)

if __name__ == "__main__":
    main()