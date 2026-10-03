import { AxiosError } from "axios"

function extractErrorMessage(err: Error): string {
  if (err instanceof AxiosError) {
    const errDetail = (err.response?.data as any)?.detail
    if (Array.isArray(errDetail) && errDetail.length > 0) {
      return errDetail[0].msg
    }
    if (typeof errDetail === "string") {
      return errDetail
    }
    return err.message
  }
  return "Something went wrong."
}

export const handleError = function (this: (msg: string) => void, err: Error) {
  const errorMessage = extractErrorMessage(err)
  this(errorMessage)
}

export const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`
}

// Determine API base URL: runtime config > build-time env > current origin.
export const getApiUrl = (): string => {
  if (typeof window !== "undefined") {
    const runtime = (window as any).APP_CONFIG?.API_URL
    if (runtime) return runtime as string
  }
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL as string
  }
  return typeof window !== "undefined" ? window.location.origin : ""
}
