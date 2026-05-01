/**
 * Scope-walking helpers shared by every language analyzer.
 *
 * The Python equivalents live in `util_ast_python.py` (`enclosing_class_node`,
 * `methods_directly_under`). The functions here are thin wrappers around the
 * generic ancestor-walk in `parser/treeUtils.ts` parameterised by language-
 * specific node-type sets.
 */

import Parser from "tree-sitter";
import { findAncestor, isSameNode, iterDescendants } from "../../parser/treeUtils";

export interface ScopeNodeTypes {
  functionLike: ReadonlySet<string>;
  classLike: ReadonlySet<string>;
}

/** Nearest enclosing function-like ancestor (inclusive), or null. */
export function enclosingFunction(
  node: Parser.SyntaxNode | null,
  types: ScopeNodeTypes
): Parser.SyntaxNode | null {
  return findAncestor(node, (n) => types.functionLike.has(n.type));
}

/** Nearest enclosing class-like ancestor (inclusive), or null. */
export function enclosingClass(
  node: Parser.SyntaxNode | null,
  types: ScopeNodeTypes
): Parser.SyntaxNode | null {
  return findAncestor(node, (n) => types.classLike.has(n.type));
}

/** Function-like nodes whose direct enclosing class is exactly `classNode`. */
export function methodsDirectlyUnder(
  classNode: Parser.SyntaxNode,
  types: ScopeNodeTypes
): Parser.SyntaxNode[] {
  const out: Parser.SyntaxNode[] = [];
  for (const n of iterDescendants(classNode)) {
    if (!types.functionLike.has(n.type)) {
      continue;
    }
    const owner = enclosingClass(n, types);
    if (isSameNode(owner, classNode)) {
      out.push(n);
    }
  }
  return out;
}

/** All function-like nodes anywhere inside `root`. */
export function allFunctions(root: Parser.SyntaxNode, types: ScopeNodeTypes): Parser.SyntaxNode[] {
  const out: Parser.SyntaxNode[] = [];
  for (const n of iterDescendants(root)) {
    if (types.functionLike.has(n.type)) {
      out.push(n);
    }
  }
  return out;
}
