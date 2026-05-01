/**
 * Identifier read/write classification, language-agnostic surface.
 *
 * The Python implementation lives in `pythonAnalyzer.ts`; this module just
 * defines the data shapes shared across analyzers (Python today, Java
 * tomorrow).
 */

import Parser from "tree-sitter";

export interface ReadWrite {
  /** True if the identifier is consumed (right-hand-side / argument / index). */
  isRead: boolean;
  /** True if the identifier is overwritten (lhs of `=`, augmented, loop var). */
  isWrite: boolean;
}

export interface IdentifierUse extends ReadWrite {
  name: string;
  /** 1-indexed line where the use occurs. */
  line: number;
  /** Byte offset of the identifier (start). */
  startByte: number;
  /** Tree-sitter node, kept for callers that need to inspect ancestors. */
  node: Parser.SyntaxNode;
}

/** Strategy interface every language analyzer must implement. */
export interface IdentifierClassifier {
  classify(node: Parser.SyntaxNode): ReadWrite;
}
