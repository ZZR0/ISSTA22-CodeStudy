/**
 * Tree-sitter parser factory.
 *
 * Centralises grammar selection (Python / Java) so the rest of the codebase
 * does not need to know which language module to load. Parsers are cached
 * per language since `Parser.setLanguage` is idempotent and constructing a
 * new parser per call is wasteful.
 */

import * as vscode from "vscode";
import Parser from "tree-sitter";
import Java from "tree-sitter-java";
import Python from "tree-sitter-python";

export type SupportedLanguage = "python" | "java";

const cache: Partial<Record<SupportedLanguage, Parser>> = {};

function parseText(parser: Parser, source: string): Parser.Tree {
  return parser.parse(source, undefined, {
    bufferSize: Math.max(32 * 1024, Buffer.byteLength(source, "utf8") + 1),
  });
}

/** Return the canonical language id used by this extension, or null. */
export function detectLanguage(document: vscode.TextDocument): SupportedLanguage | null {
  if (document.languageId === "python" || document.fileName.endsWith(".py")) {
    return "python";
  }
  if (document.languageId === "java" || document.fileName.endsWith(".java")) {
    return "java";
  }
  return null;
}

/** Tree-sitter parser pre-configured for the requested language. */
export function getParserForLanguage(language: SupportedLanguage): Parser {
  let parser = cache[language];
  if (!parser) {
    parser = new Parser();
    if (language === "python") {
      parser.setLanguage(Python as unknown as object);
    } else {
      parser.setLanguage(Java as unknown as object);
    }
    cache[language] = parser;
  }
  return parser;
}

/** Tree-sitter parser for `document`, or null when the language is unsupported. */
export function getParserForDocument(document: vscode.TextDocument): Parser | null {
  const lang = detectLanguage(document);
  return lang ? getParserForLanguage(lang) : null;
}

/** Convenience: parse the full text of a document and return root + tree. */
export interface ParsedSource {
  language: SupportedLanguage;
  parser: Parser;
  tree: Parser.Tree;
  root: Parser.SyntaxNode;
  source: string;
}

export function parseDocument(document: vscode.TextDocument): ParsedSource | null {
  const lang = detectLanguage(document);
  if (!lang) {
    return null;
  }
  const parser = getParserForLanguage(lang);
  const source = document.getText();
  const tree = parseText(parser, source);
  return { language: lang, parser, tree, root: tree.rootNode, source };
}

export function parseSource(language: SupportedLanguage, source: string): ParsedSource {
  const parser = getParserForLanguage(language);
  const tree = parseText(parser, source);
  return { language, parser, tree, root: tree.rootNode, source };
}
