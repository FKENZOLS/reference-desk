import { BookOpenText, Gauge, Library, PanelLeft, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { type ReactNode, useState } from "react"

const links = [
  { to: "/", label: "Search", icon: Search, end: true },
  { to: "/documents", label: "Documents", icon: Library },
  { to: "/workspace", label: "Workspace", icon: BookOpenText },
  { to: "/quality", label: "Quality", icon: Gauge },
]

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const path = window.location.pathname.replace(/\/+$/, "") || "/"
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[232px_minmax(0,1fr)]">
      <aside className={cn(
        "fixed inset-y-0 left-0 z-40 w-[232px] border-r bg-background/95 p-4 backdrop-blur transition-transform lg:sticky lg:top-0 lg:h-screen lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full",
      )}>
        <div className="mb-7 flex h-11 items-center gap-3 px-2">
          <div className="grid size-9 place-items-center rounded-xl bg-primary/15 text-primary">
            <BookOpenText className="size-5" />
          </div>
          <div>
            <p className="m-0 font-semibold tracking-tight">Reference Desk</p>
            <p className="m-0 text-[11px] text-muted-foreground">Local document library</p>
          </div>
        </div>
        <nav className="space-y-1">
          {links.map(({ to, label, icon: Icon }) => (
            <a
              key={to}
              href={to}
              onClick={() => setOpen(false)}
              className={cn(
                "flex h-10 items-center gap-3 rounded-lg px-3 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                path === to && "bg-primary/12 font-medium text-primary",
              )}
            >
              <Icon className="size-4" /> {label}
            </a>
          ))}
        </nav>
        <p className="absolute bottom-5 left-6 right-6 text-[11px] leading-5 text-muted-foreground">
          Your PDFs and research data stay on this computer.
        </p>
      </aside>
      {open && <button className="fixed inset-0 z-30 bg-black/55 lg:hidden" onClick={() => setOpen(false)} aria-label="Close navigation" />}
      <main className="min-w-0">
        <header className="sticky top-0 z-20 flex h-16 items-center border-b bg-background/82 px-4 backdrop-blur-xl lg:hidden">
          <Button variant="ghost" size="icon" onClick={() => setOpen(true)} aria-label="Open navigation"><PanelLeft /></Button>
          <span className="ml-2 font-semibold">Reference Desk</span>
        </header>
        <div className="mx-auto w-full max-w-[1500px] p-4 md:p-7 lg:p-9">
          {children}
        </div>
      </main>
    </div>
  )
}
