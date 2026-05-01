/**
 * High-level Python AST analyzer used by the Extract Method refactor.
 *
 * Translates the helpers in `automate_extract_method_refactoring_py/util_ast_python.py`
 * into TypeScript:
 *
 *   - `collectParams`           ↔ `collect_params`
 *   - `collectLocalVars`        ↔ `collect_local_vars`
 *   - `collectClassFields`      ↔ `collect_class_fields`
 *   - `classifyIdentifierRw`    ↔ `classify_identifier_rw`
 *   - `analyzeSelection`        ↔ `extract_rw_by_region`
 *
 * It also exposes a few surface helpers (`isStaticMethod`, `hasSelfParameter`,
 * `containsReturn`) that the orchestrator needs to make decisions about
 * the new method's signature and call site.
 */

import Parser from "tree-sitter";
import {
  IdentifierUse,
  ReadWrite,
} from "../common/identifiers";
import {
  ReadWriteRegions,
  buildRegions,
  inferInputs,
  inferOutputs,
} from "../common/dataFlow";
import {
  enclosingClass,
  enclosingFunction,
} from "../common/scopeAnalysis";
import { isSameNode, iterDescendants, textOf } from "../../parser/treeUtils";
import {
  PYTHON_PARAMETER_TYPES,
  PYTHON_SCOPE_BOUNDARIES,
  PYTHON_SCOPE_TYPES,
} from "./pythonNodeTypes";

// ---------------------------------------------------------------------------
// Function / class introspection
// ---------------------------------------------------------------------------

export interface PythonParameter {
  name: string;
  typeAnnotation?: string;
  defaultValue?: string;
  raw: string;
}

/** Return parameters declared on a `function_definition` node. */
export function collectParams(source: string, fn: Parser.SyntaxNode): PythonParameter[] {
  const paramsParent = fn.childForFieldName("parameters");
  if (!paramsParent) {
    return [];
  }

  const out: PythonParameter[] = [];
  for (const child of paramsParent.children) {
    if (!PYTHON_PARAMETER_TYPES.has(child.type)) {
      continue;
    }
    const raw = textOf(source, child).trim();
    out.push(parsePythonParameter(child, source, raw));
  }
  return out;
}

function parsePythonParameter(node: Parser.SyntaxNode, source: string, raw: string): PythonParameter {
  switch (node.type) {
    case "identifier":
      return { name: raw, raw };
    case "typed_parameter": {
      // children: identifier, ":", type
      const idNode = node.children.find((c) => c.type === "identifier");
      const typeNode = node.childForFieldName("type") ?? node.children[node.children.length - 1];
      return {
        name: idNode ? textOf(source, idNode) : raw.split(":")[0].trim(),
        typeAnnotation: typeNode ? textOf(source, typeNode) : undefined,
        raw,
      };
    }
    case "default_parameter": {
      const nameNode = node.childForFieldName("name");
      const valueNode = node.childForFieldName("value");
      return {
        name: nameNode ? textOf(source, nameNode) : raw.split("=")[0].trim(),
        defaultValue: valueNode ? textOf(source, valueNode) : undefined,
        raw,
      };
    }
    case "typed_default_parameter": {
      const nameNode = node.childForFieldName("name");
      const typeNode = node.childForFieldName("type");
      const valueNode = node.childForFieldName("value");
      return {
        name: nameNode ? textOf(source, nameNode) : raw.split(":")[0].split("=")[0].trim(),
        typeAnnotation: typeNode ? textOf(source, typeNode) : undefined,
        defaultValue: valueNode ? textOf(source, valueNode) : undefined,
        raw,
      };
    }
    case "list_splat_pattern":
    case "dictionary_splat_pattern": {
      const idNode = node.children.find((c) => c.type === "identifier");
      const cleaned = idNode ? textOf(source, idNode) : raw.replace(/^\*+/, "");
      return { name: cleaned, raw };
    }
    default:
      return { name: raw, raw };
  }
}

/** True when the function node is annotated with `@staticmethod`. */
export function isStaticMethod(fn: Parser.SyntaxNode): boolean {
  const parent = fn.parent;
  if (!parent || parent.type !== "decorated_definition") {
    return false;
  }
  for (const child of parent.children) {
    if (child.type !== "decorator") {
      continue;
    }
    const text = child.text.replace(/\s+/g, "");
    if (text.startsWith("@staticmethod")) {
      return true;
    }
  }
  return false;
}

/** True when the function declares a `self` parameter (instance method). */
export function hasSelfParameter(source: string, fn: Parser.SyntaxNode): boolean {
  const params = collectParams(source, fn);
  return params.length > 0 && params[0].name === "self";
}

/** True when the function is `async def`. */
export function isAsyncFunction(source: string, fn: Parser.SyntaxNode): boolean {
  // `async def f(...)` parses with the literal `async` token preceding `def`.
  for (const child of fn.children) {
    if (child.type === "async") {
      return true;
    }
    if (child.type === "def") {
      break;
    }
  }
  // Fallback: inspect the first source token of the definition.
  const head = source.slice(fn.startIndex, fn.startIndex + 16);
  return /^\s*async\s+def\b/.test(head);
}

/** Function name (as written in source) for a `function_definition` node. */
export function functionName(source: string, fn: Parser.SyntaxNode): string {
  const nm = fn.childForFieldName("name");
  return nm ? textOf(source, nm) : "";
}

// ---------------------------------------------------------------------------
// Local-scope discovery
// ---------------------------------------------------------------------------

export interface PythonLocalVar {
  name: string;
  typeAnnotation: string;
  line: number; // 1-indexed
}

/** Local variables defined inside `fn` (does not recurse into nested defs). */
export function collectLocalVars(source: string, fn: Parser.SyntaxNode): PythonLocalVar[] {
  const body = fn.childForFieldName("body");
  if (!body) {
    return [];
  }

  const out: PythonLocalVar[] = [];
  const stack: Parser.SyntaxNode[] = [body];

  while (stack.length > 0) {
    const node = stack.pop()!;
    const line = node.startPosition.row + 1;

    if (node.type === "assignment") {
      const left = node.childForFieldName("left");
      if (left && left.type === "identifier") {
        out.push({ name: textOf(source, left).trim(), typeAnnotation: "Any", line });
      } else if (left && (left.type === "tuple_pattern" || left.type === "pattern_list")) {
        for (const c of left.namedChildren) {
          if (c.type === "identifier") {
            out.push({ name: textOf(source, c).trim(), typeAnnotation: "Any", line });
          }
        }
      }
    } else if (node.type === "typed_assignment") {
      const left = node.childForFieldName("left");
      const type = node.childForFieldName("type");
      if (left && left.type === "identifier") {
        out.push({
          name: textOf(source, left).trim(),
          typeAnnotation: type ? textOf(source, type).trim() : "Any",
          line,
        });
      }
    } else if (node.type === "for_statement") {
      const left = node.childForFieldName("left");
      if (left && left.type === "identifier") {
        out.push({ name: textOf(source, left).trim(), typeAnnotation: "Any", line });
      }
    } else if (node.type === "except_clause") {
      const nameNode = node.childForFieldName("name");
      const typeNode = node.childForFieldName("type");
      if (nameNode) {
        out.push({
          name: textOf(source, nameNode).trim(),
          typeAnnotation: typeNode ? textOf(source, typeNode).trim() : "Exception",
          line,
        });
      }
    }

    for (let i = node.children.length - 1; i >= 0; i--) {
      const child = node.children[i];
      if (!PYTHON_SCOPE_BOUNDARIES.has(child.type)) {
        stack.push(child);
      }
    }
  }

  return out;
}

export interface PythonClassField {
  name: string;
  typeAnnotation: string;
}

/** Class-level attributes (excluding ALL_CAPS constants and `LOG`/`LOGGER`). */
export function collectClassFields(source: string, classNode: Parser.SyntaxNode | null): PythonClassField[] {
  if (!classNode) {
    return [];
  }
  const body = classNode.childForFieldName("body");
  if (!body) {
    return [];
  }

  const fields: PythonClassField[] = [];
  for (const child of body.children) {
    if (child.type !== "expression_statement" || child.children.length === 0) {
      continue;
    }
    const expr = child.children[0];
    if (expr.type !== "assignment" && expr.type !== "typed_assignment") {
      continue;
    }
    const left = expr.childForFieldName("left");
    if (!left || left.type !== "identifier") {
      continue;
    }
    const name = textOf(source, left).trim();
    let typeAnnotation = "Any";
    if (expr.type === "typed_assignment") {
      const typeNode = expr.childForFieldName("type");
      if (typeNode) {
        typeAnnotation = textOf(source, typeNode).trim();
      }
    }
    if (name.toUpperCase() === "LOG" || name.toUpperCase() === "LOGGER") {
      continue;
    }
    if (/^[A-Z0-9_]+$/.test(name)) {
      continue;
    }
    fields.push({ name, typeAnnotation });
  }
  return fields;
}

// ---------------------------------------------------------------------------
// Identifier read/write classification
// ---------------------------------------------------------------------------

/** Translate `classify_identifier_rw` from the Python helper. */
export function classifyIdentifierRw(node: Parser.SyntaxNode): ReadWrite {
  let isRead = true;
  let isWrite = false;

  if (!node.parent) {
    return { isRead, isWrite };
  }

  // Walk up through attribute / subscript wrappers (e.g. `arr[i]`, `obj.x`).
  let cur: Parser.SyntaxNode = node;
  while (cur.parent && (cur.parent.type === "attribute" || cur.parent.type === "subscript")) {
    const p = cur.parent;
    if (p.type === "subscript" && p.children[p.children.length - 2] === cur) {
      // Index/slice position — treat as ordinary read of the index.
      break;
    }
    if (p.type === "attribute" && p.childForFieldName("object") === cur) {
      break;
    }
    cur = p;
  }

  let isPureAssign = false;
  const parent = cur.parent;

  if (parent) {
    if (parent.type === "assignment" || parent.type === "typed_assignment") {
      const left = parent.childForFieldName("left");
      if (isSameNode(left, cur)) {
        isWrite = true;
        isPureAssign = true;
      } else if (left && (left.type === "pattern_list" || left.type === "tuple_pattern")) {
        if (left.children.some((child) => isSameNode(child, cur))) {
          isWrite = true;
          isPureAssign = true;
        }
      }
    } else if (parent.type === "augmented_assignment" && isSameNode(parent.childForFieldName("left"), cur)) {
      isWrite = true;
      isRead = true; // x += 1 reads the old value
    } else if (parent.type === "for_statement" && isSameNode(parent.childForFieldName("left"), cur)) {
      isWrite = true;
      isPureAssign = true;
    } else if (parent.type === "as_pattern" || parent.type === "as_pattern_target") {
      // `with ... as x:` / `except ... as x:` binds x.
      isWrite = true;
      isPureAssign = true;
    } else if (parent.type === "parameters" || parent.type === "lambda_parameters") {
      isWrite = true;
      isPureAssign = true;
    } else if (parent.type === "named_expression" && isSameNode(parent.childForFieldName("name"), cur)) {
      // Walrus operator: x := expr
      isWrite = true;
      isRead = false;
    }
  }

  if (isWrite) {
    if (isPureAssign && cur === node) {
      isRead = false;
    } else {
      isRead = true;
    }
  }

  return { isRead, isWrite };
}

// ---------------------------------------------------------------------------
// Selection analysis
// ---------------------------------------------------------------------------

export interface SelectionAnalysis {
  enclosingFunction: Parser.SyntaxNode;
  enclosingClass: Parser.SyntaxNode | null;
  isStatic: boolean;
  isInstanceMethod: boolean;
  isAsync: boolean;
  inputs: string[];
  outputs: string[];
  /** Best-effort type hints keyed by identifier name. */
  typeHints: Record<string, string>;
  /** Region report — useful for diagnostics. */
  regions: ReadWriteRegions;
  /** Distinct identifier uses observed inside the function. */
  uses: IdentifierUse[];
}

export interface AnalyzeSelectionOptions {
  /** Tree-sitter source string used to compute identifier text. */
  source: string;
  /** Function node enclosing the selection. */
  fn: Parser.SyntaxNode;
  /** 1-indexed inclusive line range of the selection within the file. */
  withinStartLine: number;
  withinEndLine: number;
}

/** Entry point: compute inputs/outputs/etc. for a selection inside a function. */
export function analyzeSelection(opts: AnalyzeSelectionOptions): SelectionAnalysis {
  const { source, fn, withinStartLine, withinEndLine } = opts;

  const params = collectParams(source, fn);
  const paramNames = new Set(params.map((p) => stripParamSplats(p.name)).filter(Boolean));

  const locals = collectLocalVars(source, fn);
  const localNames = new Set(locals.map((l) => l.name));
  const typeHints: Record<string, string> = {};
  for (const local of locals) {
    if (local.typeAnnotation && local.typeAnnotation !== "Any") {
      typeHints[local.name] = local.typeAnnotation;
    }
  }
  for (const p of params) {
    if (p.typeAnnotation) {
      typeHints[p.name] = p.typeAnnotation;
    }
  }

  const cls = enclosingClass(fn, PYTHON_SCOPE_TYPES);
  const fields = collectClassFields(source, cls);
  const fieldNames = new Set(fields.map((f) => f.name));
  for (const f of fields) {
    if (f.typeAnnotation && f.typeAnnotation !== "Any") {
      typeHints[f.name] = f.typeAnnotation;
    }
  }

  const scopeVars = new Set<string>([...paramNames, ...localNames, ...fieldNames]);

  const uses: IdentifierUse[] = [];
  for (const node of iterDescendants(fn)) {
    if (node.type !== "identifier") {
      continue;
    }
    const name = textOf(source, node);
    if (!scopeVars.has(name)) {
      continue;
    }
    const rw = classifyIdentifierRw(node);
    uses.push({
      name,
      line: node.startPosition.row + 1,
      startByte: node.startIndex,
      node,
      isRead: rw.isRead,
      isWrite: rw.isWrite,
    });
  }

  const regions = buildRegions(uses, withinStartLine, withinEndLine);
  const inputs = inferInputs(regions);
  const outputs = inferOutputs(regions);

  const isStatic = isStaticMethod(fn);
  const isInstanceMethod = !isStatic && hasSelfParameter(source, fn) && cls !== null;
  const isAsync = isAsyncFunction(source, fn);

  return {
    enclosingFunction: fn,
    enclosingClass: cls,
    isStatic,
    isInstanceMethod,
    isAsync,
    inputs,
    outputs,
    typeHints,
    regions,
    uses,
  };
}

function stripParamSplats(name: string): string {
  return name.replace(/^\*+/, "");
}

/** Locate the innermost `function_definition` containing a 0-indexed row. */
export function findEnclosingFunctionAt(
  root: Parser.SyntaxNode,
  byteOffset: number
): Parser.SyntaxNode | null {
  const node = root.descendantForIndex(byteOffset);
  return enclosingFunction(node, PYTHON_SCOPE_TYPES);
}
