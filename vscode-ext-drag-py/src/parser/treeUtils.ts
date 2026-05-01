/**
 * Generic tree-sitter walking utilities.
 *
 * Translates the cursor-based traversal helpers in `util_ast_python.py`
 * (`iter_descendants`, `iter_descendants_cursor`, `same_node`) and adds a few
 * convenience helpers used across language analyzers.
 */

import Parser from "tree-sitter";

/** Depth-first descendant iterator (left-to-right) implemented with a TreeCursor. */
export function* iterDescendants(node: Parser.SyntaxNode): Generator<Parser.SyntaxNode> {
  const cursor = node.walk();
  let done = false;

  while (!done) {
    yield cursor.currentNode;

    if (cursor.gotoFirstChild()) {
      continue;
    }
    if (cursor.gotoNextSibling()) {
      continue;
    }

    while (true) {
      if (!cursor.gotoParent()) {
        done = true;
        break;
      }
      if (cursor.gotoNextSibling()) {
        break;
      }
    }
  }
}

/** Compare nodes by `(type, span)` since tree-sitter does not guarantee identity. */
export function isSameNode(a: Parser.SyntaxNode | null, b: Parser.SyntaxNode | null): boolean {
  if (a === b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  return a.type === b.type && a.startIndex === b.startIndex && a.endIndex === b.endIndex;
}

/** Walk upward and return the nearest ancestor (inclusive) matching `predicate`. */
export function findAncestor(
  node: Parser.SyntaxNode | null,
  predicate: (n: Parser.SyntaxNode) => boolean
): Parser.SyntaxNode | null {
  let current: Parser.SyntaxNode | null = node;
  while (current) {
    if (predicate(current)) {
      return current;
    }
    current = current.parent;
  }
  return null;
}

/** Smallest node containing the half-open byte range `[startByte, endByte)`. */
export function nodeForByteRange(
  root: Parser.SyntaxNode,
  startByte: number,
  endByte: number
): Parser.SyntaxNode {
  if (endByte <= startByte) {
    return root.descendantForIndex(startByte);
  }
  return root.descendantForIndex(startByte, endByte - 1);
}

/** Substring of `source` covered by `node` (UTF-16 — VSCode's native string view). */
export function textOf(source: string, node: Parser.SyntaxNode): string {
  return source.slice(node.startIndex, node.endIndex);
}

/** Innermost named descendant containing `byteOffset`. Falls back to `root`. */
export function innermostNamedAt(root: Parser.SyntaxNode, byteOffset: number): Parser.SyntaxNode {
  return root.namedDescendantForIndex(byteOffset);
}
