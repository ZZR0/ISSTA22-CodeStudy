/**
 * Tiny logging facade so the extension can emit structured trace messages
 * without sprinkling `console.log` everywhere.
 *
 * Use `logger.info`/`logger.warn`/`logger.error` for operational messages and
 * `logger.debug` for verbose tracing that is silenced by default.
 */

let debugEnabled = false;

export function setDebugEnabled(value: boolean): void {
  debugEnabled = value;
}

function fmt(scope: string, level: string, parts: unknown[]): string {
  const stamp = new Date().toISOString().slice(11, 23);
  return `[${stamp}] [${level}] [${scope}] ${parts
    .map((p) => (typeof p === "string" ? p : JSON.stringify(p)))
    .join(" ")}`;
}

export const logger = {
  info(scope: string, ...parts: unknown[]): void {
    console.log(fmt(scope, "INFO", parts));
  },
  warn(scope: string, ...parts: unknown[]): void {
    console.warn(fmt(scope, "WARN", parts));
  },
  error(scope: string, ...parts: unknown[]): void {
    console.error(fmt(scope, "ERROR", parts));
  },
  debug(scope: string, ...parts: unknown[]): void {
    if (debugEnabled) {
      console.log(fmt(scope, "DBG", parts));
    }
  },
};
