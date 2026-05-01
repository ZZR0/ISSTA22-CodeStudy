import * as vscode from "vscode";

import Parser from "tree-sitter";
import Java from "tree-sitter-java";
import Python from "tree-sitter-python";

function getParserForDocument(document: vscode.TextDocument): Parser | null {
  const parser = new Parser();

  if (document.languageId === "java" || document.fileName.endsWith(".java")) {
    parser.setLanguage(Java);
    return parser;
  }

  if (document.languageId === "python" || document.fileName.endsWith(".py")) {
    parser.setLanguage(Python);
    return parser;
  }

  return null;
}

function isFunctionLikeNode(type: string): boolean {
  return (
    type === "method_declaration" ||
    type === "constructor_declaration" ||
    type === "function_definition"
  );
}

function parseDocumentText(parser: Parser, sourceCode: string): Parser.Tree {
  return parser.parse(sourceCode, undefined, {
    bufferSize: Math.max(32 * 1024, Buffer.byteLength(sourceCode, "utf8") + 1),
  });
}

export async function getEnclosingFunctionViaTreeSitter(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<string> {
  const parser = getParserForDocument(document);

  if (!parser) {
    return "<unsupported language>";
  }

  const sourceCode = document.getText();
  const tree = parseDocumentText(parser, sourceCode);

  const offset = document.offsetAt(position);
  const node = tree.rootNode.descendantForIndex(offset);

  let current = node;

  while (current) {
    if (isFunctionLikeNode(current.type)) {
      return current.text;
    }

    if (!current.parent) {
      break;
    }

    current = current.parent;
  }

  return "<no enclosing function found>";
}

export async function getEnclosingFunctionWithPosition(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<{ text: string; startPosition: vscode.Position } | null> {
  const parser = getParserForDocument(document);

  if (!parser) {
    return null;
  }

  const sourceCode = document.getText();
  const tree = parseDocumentText(parser, sourceCode);

  const offset = document.offsetAt(position);
  const node = tree.rootNode.descendantForIndex(offset);

  let current = node;

  while (current) {
    if (isFunctionLikeNode(current.type)) {
      return {
        text: current.text,
        startPosition: document.positionAt(current.startIndex),
      };
    }

    if (!current.parent) {
      break;
    }

    current = current.parent;
  }

  return null;
}
