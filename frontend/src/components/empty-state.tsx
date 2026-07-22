import type { LucideIcon } from "lucide-react"

export function EmptyState({ icon: Icon, title, detail }: { icon?: LucideIcon; title: string; detail?: string }) {
  return (
    <div className="grid min-h-56 place-items-center rounded-xl border border-dashed bg-card/35 p-8 text-center">
      <div>
        {Icon && <Icon className="mx-auto mb-3 size-7 text-muted-foreground" />}
        <p className="m-0 font-medium">{title}</p>
        {detail && <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">{detail}</p>}
      </div>
    </div>
  )
}
