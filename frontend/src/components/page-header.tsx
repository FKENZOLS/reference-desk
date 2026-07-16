import type { ReactNode } from "react"

export function PageHeader({ title, actions }: { title: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex min-h-11 flex-wrap items-center justify-between gap-3">
      <h1 className="m-0 text-2xl font-semibold tracking-tight md:text-[28px]">{title}</h1>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}
