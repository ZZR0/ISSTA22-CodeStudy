import * as vscode from "vscode";

import Parser from "tree-sitter";
import Java from "tree-sitter-java";
import Python from "tree-sitter-python";

type EnclosingFunctionResult = {
  text: string;
  startPosition: vscode.Position;
};

function getParserForDocument(document: vscode.TextDocument): Parser | null {
  const parser = new Parser();

  if (document.languageId === "python" || document.fileName.endsWith(".py")) {
    parser.setLanguage(Python);
    return parser;
  }

  if (document.languageId === "java" || document.fileName.endsWith(".java")) {
    parser.setLanguage(Java);
    return parser;
  }

  return null;
}

function isFunctionLikeNode(type: string): boolean {
  return (
    type === "function_definition" ||
    type === "method_declaration" ||
    type === "constructor_declaration"
  );
}

function parseDocumentText(parser: Parser, sourceCode: string): Parser.Tree {
  return parser.parse(sourceCode, undefined, {
    bufferSize: Math.max(32 * 1024, Buffer.byteLength(sourceCode, "utf8") + 1),
  });
}

function findEnclosingFunctionNode(
  document: vscode.TextDocument,
  position: vscode.Position
): Parser.SyntaxNode | null {
  const parser = getParserForDocument(document);

  if (!parser) {
    return null;
  }

  const sourceCode = document.getText();
  const tree = parseDocumentText(parser, sourceCode);
  const node = tree.rootNode.descendantForIndex(document.offsetAt(position));

  let current: Parser.SyntaxNode | null = node;
  while (current) {
    if (isFunctionLikeNode(current.type)) {
      return current;
    }

    current = current.parent;
  }

  return null;
}

export async function getEnclosingFunctionViaTreeSitter(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<string> {
  const node = findEnclosingFunctionNode(document, position);
  return node?.text ?? "<no enclosing function found>";
}

export async function getEnclosingFunctionWithPosition(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<EnclosingFunctionResult | null> {
  const node = findEnclosingFunctionNode(document, position);

  if (!node) {
    return null;
  }

  return {
    text: node.text,
    startPosition: document.positionAt(node.startIndex),
  };
}

export async function getEnclosingMethodViaTreeSitter(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<string> {
  return getEnclosingFunctionViaTreeSitter(document, position);
}

export async function getEnclosingMethodWithPosition(
  document: vscode.TextDocument,
  position: vscode.Position
): Promise<EnclosingFunctionResult | null> {
  return getEnclosingFunctionWithPosition(document, position);
}
