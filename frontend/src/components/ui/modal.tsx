import { X } from "lucide-react"
import type { ReactNode } from "react"
import { Button } from "@/components/ui/button"

export function Modal({ open, title, children, onClose, wide = false }: { open: boolean; title: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className={`max-h-[88vh] w-full overflow-auto rounded-xl border bg-popover shadow-2xl ${wide ? "max-w-5xl" : "max-w-xl"}`}>
        <div className="sticky top-0 z-10 flex items-center justify-between border-b bg-popover/95 px-5 py-4 backdrop-blur">
          <h2 className="m-0 text-base font-semibold">{title}</h2>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close"><X /></Button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  )
}
