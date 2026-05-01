/**
 * Detection of Python constructs that make a selection ineligible for
 * Extract Method.
 *
 * The VSCode built-in refactor refuses to extract code that contains:
 *
 *   - `return` (lifts control flow out of the original method)
 *   - `yield` / `yield from`
 *   - `await` when the host method is not `async`
 *   - `break` / `continue` whose target loop is outside the selection
 *   - `nonlocal` / `global` declarations (rebinding outer scopes)
 *
 * The analyzer reports any of these so the orchestrator can refuse the
 * refactor with an actionable diagnostic.
 */

import Parser from "tree-sitter";
import { isSameNode, iterDescendants } from "../../parser/treeUtils";

export type UnsupportedKind =
  | "return"
  | "yield"
  | "await-in-sync"
  | "break-outside-loop"
  | "continue-outside-loop"
  | "nonlocal"
  | "global"
  | "syntax-error";

export interface UnsupportedFinding {
  kind: UnsupportedKind;
  startLine: number; // 0-indexed (tree-sitter row)
  endLine: number;
  text: string;
}

interface SelectionContext {
  enclosingFunctionIsAsync: boolean;
  containsLoop: boolean;
  /** Byte range of the user's selection (start inclusive, end exclusive). */
  selectionStartByte: number;
  selectionEndByte: number;
}

function nodeIsAncestor(maybeAncestor: Parser.SyntaxNode, node: Parser.SyntaxNode): boolean {
  let cur: Parser.SyntaxNode | null = node.parent;
  while (cur) {
    if (isSameNode(cur, maybeAncestor)) {
      return true;
    }
    cur = cur.parent;
  }
  return false;
}

/** True when `node` is fully (or majorly) inside `[selStart, selEnd)`. */
function nodeIsInsideSelection(
  node: Parser.SyntaxNode,
  selStart: number,
  selEnd: number
): boolean {
  return node.startIndex >= selStart && node.endIndex <= selEnd;
}

function isInsideLoopWithinSelection(
  node: Parser.SyntaxNode,
  selStart: number,
  selEnd: number
): boolean {
  let cur: Parser.SyntaxNode | null = node.parent;
  while (cur) {
    if (cur.type === "for_statement" || cur.type === "while_statement") {
      // The loop itself has to live inside the user's selection for the
      // break/continue to remain self-contained after extraction.
      if (nodeIsInsideSelection(cur, selStart, selEnd)) {
        return true;
      }
    }
    cur = cur.parent;
  }
  return false;
}

/**
 * Scan a selection root and return any constructs that prevent extraction.
 *
 * @param selectionRoot  The smallest tree-sitter node whose span equals (or
 *                       contains) the user's selection. Pass the deepest
 *                       common ancestor when extracting multiple statements.
 * @param ctx            Information about the enclosing function so we can
 *                       judge `await` correctness.
 */
export function findUnsupportedConstructs(
  selectionRoot: Parser.SyntaxNode,
  ctx: SelectionContext
): UnsupportedFinding[] {
  const findings: UnsupportedFinding[] = [];
  const { selectionStartByte: selStart, selectionEndByte: selEnd } = ctx;

  const record = (kind: UnsupportedKind, n: Parser.SyntaxNode) => {
    findings.push({
      kind,
      startLine: n.startPosition.row,
      endLine: n.endPosition.row,
      text: n.text,
    });
  };

  for (const node of iterDescendants(selectionRoot)) {
    // Tree-sitter's `selectionRoot` may be wider than the user's actual
    // selection (it returns the smallest *named* node that contains the byte
    // range, which can climb to the function body or module when the
    // selection straddles sibling statements). Only scan nodes that lie
    // strictly inside the user's selected bytes.
    if (!nodeIsInsideSelection(node, selStart, selEnd)) {
      continue;
    }

    switch (node.type) {
      case "return_statement":
        record("return", node);
        break;
      case "yield":
        record("yield", node);
        break;
      case "nonlocal_statement":
        record("nonlocal", node);
        break;
      case "global_statement":
        record("global", node);
        break;
      case "await":
      case "await_expression":
        if (!ctx.enclosingFunctionIsAsync) {
          record("await-in-sync", node);
        }
        break;
      case "break_statement":
        if (!isInsideLoopWithinSelection(node, selStart, selEnd)) {
          record("break-outside-loop", node);
        }
        break;
      case "continue_statement":
        if (!isInsideLoopWithinSelection(node, selStart, selEnd)) {
          record("continue-outside-loop", node);
        }
        break;
      default:
        if (node.isError || node.isMissing) {
          record("syntax-error", node);
        }
        break;
    }
  }

  return findings;
}
