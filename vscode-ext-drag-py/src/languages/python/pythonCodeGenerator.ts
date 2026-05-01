/**
 * Render the extracted method definition and matching call site.
 *
 * Mirrors the responsibilities of `PythonMethodRenderer` and
 * `Rewriter._build_callsite` from `extract_method_refactor_python.py`, but
 * exposed as pure functions so the orchestrator (`extractMethodService`) can
 * compose them with VSCode's edit infrastructure.
 */

import { dedentBlock, indentBlock } from "../../utils/indentation";

export interface PythonMethodSignature {
  /** Method name as written after `def`. */
  name: string;
  /** Parameter names, in source order. `self` is implicit (added by render()). */
  parameters: string[];
  /** Optional `Type name` annotations, keyed by parameter name. */
  typeHints?: Record<string, string>;
  /** Names that the method should return (empty when nothing flows out). */
  returnVariables: string[];
  /** Whether the method is decorated with `@staticmethod`. */
  isStatic: boolean;
  /** Whether the host function is `async def`. */
  isAsync: boolean;
  /** True if the host scope is inside a class. */
  inClassScope: boolean;
  /** True if the original function is an instance method (i.e. has `self`). */
  isInstanceMethod: boolean;
  /** Optional return-type annotation. */
  returnTypeHint?: string;
}

/** Render the new method as a string of Python source. */
export function renderPythonMethod(signature: PythonMethodSignature, body: string): string {
  const lines: string[] = [];

  if (signature.inClassScope && signature.isStatic) {
    lines.push("@staticmethod");
  }

  const params: string[] = [];
  if (signature.inClassScope && !signature.isStatic && signature.isInstanceMethod) {
    params.push("self");
  }
  for (const p of signature.parameters) {
    const annot = signature.typeHints?.[p];
    params.push(annot && annot !== "Any" ? `${p}: ${annot}` : p);
  }

  const asyncPrefix = signature.isAsync ? "async " : "";
  const returnSuffix = signature.returnTypeHint ? ` -> ${signature.returnTypeHint}` : "";
  lines.push(`${asyncPrefix}def ${signature.name}(${params.join(", ")})${returnSuffix}:`);

  const dedented = dedentBlock(body).replace(/\s+$/g, "");
  if (dedented.trim().length === 0) {
    lines.push("    pass");
  } else {
    lines.push(indentBlock(dedented, "    "));
  }

  if (signature.returnVariables.length > 0) {
    // Avoid emitting a redundant trailing return when the body already returns
    // a sentinel `__return__` that the orchestrator pre-injected.
    const hasInlineReturn = /\breturn\b/.test(dedented) && signature.returnVariables.includes("__return__");
    if (!hasInlineReturn) {
      lines.push(indentBlock(`return ${signature.returnVariables.join(", ")}`, "    "));
    }
  }

  return lines.join("\n");
}

export interface PythonCallSiteOptions {
  signature: PythonMethodSignature;
  /** Identifier expressions to pass as arguments, in the same order as `signature.parameters`. */
  argumentExprs: string[];
  /** LHS identifiers to receive return values; empty for no assignment. */
  outputAssignments: string[];
  /** Indentation prefix to apply to the whole call line. */
  indent: string;
}

/** Build the single-line call expression that replaces the original selection. */
export function renderPythonCallSite(opts: PythonCallSiteOptions): string {
  const { signature, argumentExprs, outputAssignments, indent } = opts;

  const callerPrefix = signature.inClassScope && !signature.isStatic && signature.isInstanceMethod
    ? "self."
    : "";

  const awaitPrefix = signature.isAsync ? "await " : "";
  const args = argumentExprs.join(", ");
  const call = `${awaitPrefix}${callerPrefix}${signature.name}(${args})`;

  const lhs = outputAssignments.length > 0 ? `${outputAssignments.join(", ")} = ` : "";
  return `${indent}${lhs}${call}\n`;
}
