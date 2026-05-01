/**
 * Tree-sitter Python node-type constants used by the analyzer.
 *
 * Centralised here so the rest of the Python module only references symbolic
 * names. Mirrors the `_METHOD_NODES` / `_CLASS_NODES` / `scope_boundaries`
 * constants from `util_ast_python.py` and `extract_method_refactor_python.py`.
 */

import { ScopeNodeTypes } from "../common/scopeAnalysis";

export const PYTHON_FUNCTION_LIKE: ReadonlySet<string> = new Set([
  "function_definition",
]);

export const PYTHON_CLASS_LIKE: ReadonlySet<string> = new Set([
  "class_definition",
]);

export const PYTHON_SCOPE_BOUNDARIES: ReadonlySet<string> = new Set([
  "function_definition",
  "class_definition",
  "lambda",
]);

export const PYTHON_PARAMETER_TYPES: ReadonlySet<string> = new Set([
  "identifier",
  "typed_parameter",
  "default_parameter",
  "list_splat_pattern",
  "dictionary_splat_pattern",
  "typed_default_parameter",
]);

export const PYTHON_SCOPE_TYPES: ScopeNodeTypes = {
  functionLike: PYTHON_FUNCTION_LIKE,
  classLike: PYTHON_CLASS_LIKE,
};
