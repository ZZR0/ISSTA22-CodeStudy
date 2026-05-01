/**
 * Decide what the new method should look like, given the analyzer's output.
 *
 * Mirrors `SignatureSynthesizer` from `extract_method_refactor_python.py` but
 * built on top of {@link analyzeSelection} rather than the per-clone JSON
 * payload — the VSCode flow only ever has one selection at a time.
 */

import { SelectionAnalysis } from "../languages/python/pythonAnalyzer";
import { sanitizeIdentifier, uniqueMethodName } from "../utils/names";
import { ExtractMethodOptions, ExtractedSignature } from "./types";

export interface InferSignatureInput {
  analysis: SelectionAnalysis;
  options?: ExtractMethodOptions;
  /** Sibling method/function names that must not collide. */
  existingNames: Iterable<string>;
}

export function inferSignature(input: InferSignatureInput): ExtractedSignature {
  const { analysis, options, existingNames } = input;

  const inClassScope = analysis.enclosingClass !== null;
  const isStatic = options?.forceStatic ? true : analysis.isStatic;

  // `self` is implicit in `renderPythonMethod` — exclude it from parameters.
  const parameters: string[] = [];
  for (const name of analysis.inputs) {
    if (inClassScope && !isStatic && name === "self") {
      continue;
    }
    parameters.push(sanitizeIdentifier(name));
  }

  const returnVariables = analysis.outputs.map(sanitizeIdentifier);

  const methodName = uniqueMethodName(options?.methodName ?? "extracted", existingNames);

  const typeHints: Record<string, string> = {};
  for (const name of parameters) {
    const hint = analysis.typeHints[name];
    if (hint && hint !== "Any") {
      typeHints[name] = hint;
    }
  }

  return {
    methodName,
    parameters,
    returnVariables,
    isStatic,
    isAsync: analysis.isAsync,
    inClassScope,
    isInstanceMethod: analysis.isInstanceMethod,
    typeHints,
    returnTypeHint: options?.returnTypeHint,
  };
}
