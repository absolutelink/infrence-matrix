import { AxiosError } from "axios"

// Extract a human-readable message from an API error. The admin surfaces
// errors as {detail: string | {error, step} | ValidationError[]}.
export function extractError(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = (err.response?.data as Record<string, unknown> | undefined)
      ?.detail
    if (typeof detail === "string") return detail
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as Record<string, unknown>
      return String(first.msg ?? "Validation error")
    }
    if (detail && typeof detail === "object") {
      const d = detail as Record<string, unknown>
      const step = d.step ? ` (step: ${String(d.step)})` : ""
      const retry =
        d.retry_after != null ? ` (retry in ${String(d.retry_after)}s)` : ""
      return `${String(d.error ?? "provider refused")}${step}${retry}`
    }
    return err.message
  }
  if (err instanceof Error) return err.message
  return "Something went wrong."
}
