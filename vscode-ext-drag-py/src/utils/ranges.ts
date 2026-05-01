/**
 * Line/byte range arithmetic shared by the refactor pipeline.
 *
 * Mirrors the helpers in `extract_method_refactor_python.py`
 * (`parse_range`, `line_offsets`, `line_span_to_offsets`,
 * `apply_replacements`).
 */

export interface LineSpan {
  /** 1-indexed inclusive start. */
  startLine: number;
  /** 1-indexed inclusive end. */
  endLine: number;
}

const RANGE_RE = /^(\d+)-(\d+)$/;

/** Parse a string of the form `"412-425"` into a {@link LineSpan}. */
export function parseRange(rangeText: string): LineSpan {
  const m = RANGE_RE.exec(rangeText.trim());
  if (!m) {
    throw new Error(`Invalid range: ${rangeText}`);
  }
  const startLine = Number(m[1]);
  const endLine = Number(m[2]);
  if (startLine > endLine) {
    throw new Error(`Invalid line range order: ${rangeText}`);
  }
  return { startLine, endLine };
}

/**
 * Return the byte offsets of every line start in `text`. The returned array
 * has length `lineCount + 1` so callers can compute the end-of-file offset
 * with `offsets[offsets.length - 1]`.
 */
export function lineOffsets(text: string): number[] {
  const offsets: number[] = [0];
  let running = 0;
  let cursor = 0;
  while (cursor < text.length) {
    const next = text.indexOf("\n", cursor);
    if (next === -1) {
      running += text.length - cursor;
      offsets.push(running);
      break;
    }
    running += next - cursor + 1;
    offsets.push(running);
    cursor = next + 1;
  }
  if (offsets.length === 1) {
    offsets.push(0);
  }
  return offsets;
}

/** Convert a 1-indexed line span into `[startOffset, endOffset]`. */
export function lineSpanToOffsets(text: string, span: LineSpan): [number, number] {
  const offsets = lineOffsets(text);
  if (span.endLine > offsets.length - 1) {
    throw new Error(`Range ${span.startLine}-${span.endLine} exceeds file length`);
  }
  return [offsets[span.startLine - 1], offsets[span.endLine]];
}

export interface OffsetReplacement {
  start: number;
  end: number;
  text: string;
}

/**
 * Apply non-overlapping replacements to `source`. Replacements are sorted by
 * descending start offset before splicing so earlier offsets do not shift.
 */
export function applyReplacements(source: string, replacements: OffsetReplacement[]): string {
  const sorted = [...replacements].sort((a, b) => b.start - a.start);
  let out = source;
  for (const { start, end, text } of sorted) {
    out = out.slice(0, start) + text + out.slice(end);
  }
  return out;
}
