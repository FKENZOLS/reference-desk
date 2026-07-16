export type ClassValue = string | number | null | undefined | false | ClassValue[] | Record<string, unknown>

export function clsx(...inputs: ClassValue[]): string {
  const output: string[] = []
  const visit = (value: ClassValue) => {
    if (!value) return
    if (typeof value === "string" || typeof value === "number") output.push(String(value))
    else if (Array.isArray(value)) value.forEach(visit)
    else Object.entries(value).forEach(([key, enabled]) => enabled && output.push(key))
  }
  inputs.forEach(visit)
  return output.join(" ")
}
