import { useEffect, useState } from "react"

type ToastKind = "success" | "error" | "message"
type ToastItem = { id: number; kind: ToastKind; text: string }
let items: ToastItem[] = []
let nextId = 1
const listeners = new Set<(value: ToastItem[]) => void>()

function push(kind: ToastKind, text: string) {
  const item = { id: nextId++, kind, text }
  items = [...items, item]
  listeners.forEach((listener) => listener(items))
  window.setTimeout(() => {
    items = items.filter((value) => value.id !== item.id)
    listeners.forEach((listener) => listener(items))
  }, 3600)
}

export const toast = {
  success: (text: string) => push("success", text),
  error: (text: string) => push("error", text),
  message: (text: string) => push("message", text),
}

export function Toaster(_props: Record<string, unknown>) {
  const [visible, setVisible] = useState(items)
  useEffect(() => { listeners.add(setVisible); return () => { listeners.delete(setVisible) } }, [])
  return <div className="toast-stack" aria-live="polite">{visible.map((item) => <div key={item.id} className={`toast-item toast-${item.kind}`}>{item.text}</div>)}</div>
}
