/**
 * Region-based read/write data-flow analysis.
 *
 * Mirrors `extract_rw_by_region` from `util_ast_python.py`. The Python script
 * partitions the enclosing function's lines into PRE / WITHIN / POST relative
 * to a clone span and tracks which identifiers are read or written in each
 * region. The same partitioning lets us infer:
 *
 *   - **inputs**  — variables read WITHIN that originate from PRE.
 *   - **outputs** — variables written WITHIN that are read in POST.
 */

import { IdentifierUse } from "./identifiers";

export type Region = "pre" | "within" | "post";

export interface RegionSets {
  pre: Set<string>;
  within: Set<string>;
  post: Set<string>;
}

export interface ReadWriteRegions {
  reads: RegionSets;
  writes: RegionSets;
  /** Names assigned (purely overwritten) at least once inside `within`. */
  locallyDefinedWithin: Set<string>;
}

export function emptyRegions(): ReadWriteRegions {
  return {
    reads: { pre: new Set(), within: new Set(), post: new Set() },
    writes: { pre: new Set(), within: new Set(), post: new Set() },
    locallyDefinedWithin: new Set<string>(),
  };
}

/** Map a 1-indexed line number to its region given the within-clone span. */
export function regionForLine(line: number, withinStart: number, withinEnd: number): Region {
  if (line < withinStart) {
    return "pre";
  }
  if (line > withinEnd) {
    return "post";
  }
  return "within";
}

/**
 * Given an iterable of identifier uses already classified read/write, fold
 * them into per-region sets. Caller controls which identifiers count (for
 * example "only those in the function's local scope").
 */
export function buildRegions(
  uses: Iterable<IdentifierUse>,
  withinStart: number,
  withinEnd: number,
  predicate?: (use: IdentifierUse) => boolean
): ReadWriteRegions {
  const regions = emptyRegions();

  for (const use of uses) {
    if (predicate && !predicate(use)) {
      continue;
    }
    const region = regionForLine(use.line, withinStart, withinEnd);

    if (region === "within") {
      if (use.isRead && !regions.locallyDefinedWithin.has(use.name)) {
        regions.reads.within.add(use.name);
      }
      if (use.isWrite) {
        regions.writes.within.add(use.name);
        regions.locallyDefinedWithin.add(use.name);
      }
    } else {
      if (use.isRead) {
        regions.reads[region].add(use.name);
      }
      if (use.isWrite) {
        regions.writes[region].add(use.name);
      }
    }
  }

  return regions;
}

/**
 * Names that are read WITHIN and either written PRE or never written at all
 * inside the function — the canonical "free variables" for an extract.
 */
export function inferInputs(regions: ReadWriteRegions): string[] {
  const inputs: string[] = [];
  for (const name of regions.reads.within) {
    if (regions.writes.pre.has(name) || (!regions.writes.within.has(name) && !regions.writes.post.has(name))) {
      inputs.push(name);
    } else if (regions.writes.pre.has(name)) {
      inputs.push(name);
    }
  }
  // de-dupe while preserving order
  return Array.from(new Set(inputs));
}

/** Names written WITHIN and read POST — the variables the new method must return. */
export function inferOutputs(regions: ReadWriteRegions): string[] {
  const outputs: string[] = [];
  for (const name of regions.writes.within) {
    if (regions.reads.post.has(name)) {
      outputs.push(name);
    }
  }
  return Array.from(new Set(outputs));
}
