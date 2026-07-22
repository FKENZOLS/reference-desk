import { Button } from "@/components/ui/button"
import { api } from "@/lib/api"
import { cn } from "@/lib/utils"
import type { AppNotification } from "@/types"
import { type ReactNode, useEffect, useState } from "react"

const primaryLinks = [
  { to: "/", label: "Search" },
  { to: "/documents", label: "Documents" },
  { to: "/workspace", label: "Workspace" },
]

const advancedLinks = [
  { to: "/quality", label: "Quality" },
  { to: "/experiments", label: "Experiments" },
]

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [noticesOpen, setNoticesOpen] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [notifications, setNotifications] = useState<AppNotification[]>([])
  const path = window.location.pathname.replace(/\/+$/, "") || "/"

  useEffect(() => {
    async function refreshNotifications() {
      try {
        const value = await api<{ notifications: AppNotification[] }>("/api/notifications")
        setNotifications(value.notifications)
      } catch { /* notifications must never interrupt the main workspace */ }
    }
    void refreshNotifications()
    const timer = window.setInterval(refreshNotifications, 5000)
    return () => window.clearInterval(timer)
  }, [])

  async function markRead(id: number) {
    try {
      const updated = await api<AppNotification>(`/api/notifications/${id}/read`, { method: "POST" })
      setNotifications((current) => current.map((item) => item.id === id ? updated : item))
    } catch { /* the next poll retries naturally */ }
  }

  const unread = notifications.filter((item) => !item.read_at).length
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[232px_minmax(0,1fr)]">
      <aside className={cn(
        "fixed inset-y-0 left-0 z-40 flex w-[232px] flex-col border-r bg-background/95 p-4 backdrop-blur transition-transform lg:sticky lg:top-0 lg:h-screen lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full",
      )}>
        <nav className="space-y-1">
          {primaryLinks.map(({ to, label }) => (
            <a
              key={to}
              href={to}
              onClick={() => setOpen(false)}
              className={cn(
                "flex h-10 items-center rounded-lg px-3 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                path === to && "bg-primary/12 font-medium text-primary",
              )}
            >
              {label}
            </a>
          ))}
        </nav>
        <div className="mt-auto border-t pt-4">
          <button type="button" className="flex h-10 w-full items-center rounded-lg px-3 text-left text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground" aria-expanded={advancedOpen} onClick={() => setAdvancedOpen((value) => !value)}>Advanced</button>
          {advancedOpen && <nav className="mt-1 space-y-1">{advancedLinks.map(({ to, label }) => <a key={to} href={to} onClick={() => setOpen(false)} className={cn("flex h-9 items-center rounded-lg px-5 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground", path === to && "bg-primary/12 font-medium text-primary")}>{label}</a>)}</nav>}
        </div>
      </aside>
      {open && <button className="fixed inset-0 z-30 bg-black/55 lg:hidden" onClick={() => setOpen(false)} aria-label="Close navigation" />}
      <main className="min-w-0">
        <div className="notification-anchor fixed z-50">
          <Button variant="outline" size="sm" className="relative bg-background/90 shadow-sm backdrop-blur" onClick={() => setNoticesOpen((value) => !value)} aria-label="Background notifications">Updates{unread > 0 && <span className="ml-2 grid min-w-4 place-items-center rounded-full bg-primary px-1 text-[9px] leading-4 text-primary-foreground">{Math.min(unread, 99)}</span>}</Button>
          {noticesOpen && <div className="absolute right-0 mt-2 w-[min(360px,calc(100vw-2rem))] overflow-hidden rounded-xl border bg-card shadow-2xl"><div className="border-b px-4 py-3"><p className="m-0 text-sm font-semibold">Background activity</p><p className="m-0 mt-0.5 text-[11px] text-muted-foreground">Persistent updates from models, indexing, backups, and experiments.</p></div><div className="max-h-96 overflow-y-auto">{notifications.length ? notifications.map((item) => <button key={item.id} type="button" onClick={() => void markRead(item.id)} className={cn("block w-full border-b px-4 py-3 text-left last:border-0 hover:bg-accent/50", !item.read_at && "bg-primary/5")}><div className="flex items-center justify-between gap-2"><span className="text-sm font-medium">{item.title}</span><span className={cn("size-2 shrink-0 rounded-full", item.status === "error" ? "bg-destructive" : item.status === "warning" ? "bg-amber-400" : item.status === "success" ? "bg-emerald-400" : "bg-primary")} /></div>{item.message && <p className="mb-0 mt-1 text-xs leading-5 text-muted-foreground">{item.message}</p>}<p className="mb-0 mt-1 text-[10px] text-muted-foreground">{new Date(item.created_at).toLocaleString()}</p></button>) : <p className="m-0 px-4 py-8 text-center text-sm text-muted-foreground">No background updates yet.</p>}</div></div>}
        </div>
        <header className="sticky top-0 z-20 flex h-16 items-center border-b bg-background/82 px-4 backdrop-blur-xl lg:hidden">
          <Button variant="ghost" size="sm" onClick={() => setOpen(true)} aria-label="Open navigation">Menu</Button>
        </header>
        <div className="mx-auto w-full max-w-[1500px] p-4 md:p-7 lg:p-9">
          {children}
        </div>
      </main>
    </div>
  )
}
