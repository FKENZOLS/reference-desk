import { useEffect, useState } from "react"
import { toast } from "sonner"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { api, jsonRequest } from "@/lib/api"

type UpdateState = {
  available: boolean
  can_update: boolean
  blocked_reason?: string | null
  error?: string | null
  version?: string
  branch?: string
  upstream?: string
  remote_url?: string
  local_short?: string
  remote_short?: string
  local_commit?: string
  remote_commit?: string
  latest_subject?: string
  latest_date?: string
  ahead?: number
  behind?: number
  changed_paths?: string[]
  supervised_restart: boolean
  restart_required: boolean
  restart_scheduled?: boolean
  updated?: boolean
  action_token: string
}

function shortDate(value?: string) {
  if (!value) return "—"
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export function UpdatesPage() {
  const [state, setState] = useState<UpdateState | null>(null)
  const [checking, setChecking] = useState(true)
  const [installing, setInstalling] = useState(false)

  async function check() {
    setChecking(true)
    try {
      const value = await api<UpdateState>("/updates/api/check", { method: "POST" })
      setState(value)
      if (value.error) toast.error(value.error)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not check GitHub")
    } finally {
      setChecking(false)
    }
  }

  useEffect(() => { void check() }, [])

  async function waitForRestart(expectedCommit?: string) {
    await new Promise((resolve) => window.setTimeout(resolve, 2200))
    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      try {
        const response = await fetch(`/updates/api/status?at=${Date.now()}`, { cache: "no-store" })
        if (response.ok) {
          const value = await response.json() as UpdateState
          if (!expectedCommit || value.local_commit === expectedCommit) {
            window.location.reload()
            return
          }
        }
      } catch { /* the server is expected to disappear briefly */ }
      await new Promise((resolve) => window.setTimeout(resolve, 1500))
    }
    setInstalling(false)
    toast.error("The restart is taking longer than expected. Reopen Reference Desk from START.bat.")
  }

  async function install() {
    if (!state?.can_update) return
    const action = state.supervised_restart ? "install this update and restart Reference Desk" : "install this update"
    if (!window.confirm(`Ready to ${action}? Your PDFs, indexes, settings, and workspace data will be preserved.`)) return
    setInstalling(true)
    try {
      const value = await api<UpdateState>("/updates/api/apply", jsonRequest("POST", {
        confirm: true,
        action_token: state.action_token,
      }))
      setState(value)
      if (!value.updated) {
        setInstalling(false)
        toast.success("Reference Desk is already up to date")
      } else if (value.restart_scheduled) {
        toast.success("Update installed. Reference Desk is restarting.")
        void waitForRestart(value.local_commit)
      } else {
        setInstalling(false)
        toast.success("Update installed. Restart Reference Desk to load it.")
      }
    } catch (error) {
      setInstalling(false)
      toast.error(error instanceof Error ? error.message : "Could not install the update")
      await check()
    }
  }

  const status = state?.error
    ? "Unavailable"
    : state?.blocked_reason
      ? "Action needed"
      : state?.available
        ? "Update available"
        : "Up to date"

  return (
    <div>
      <PageHeader
        title="Software updates"
        actions={<Button variant="outline" onClick={() => void check()} disabled={checking || installing}>{checking ? "Checking…" : "Check again"}</Button>}
      />
      <p className="-mt-4 mb-6 text-sm text-muted-foreground">Safely update this source installation from its configured GitHub repository.</p>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className="space-y-5">
          <Card className="bg-card/75">
            <CardContent className="p-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="m-0 text-xs font-medium uppercase tracking-wide text-muted-foreground">Update status</p>
                  <h2 className="mb-0 mt-2 text-xl font-semibold">{checking && !state ? "Checking GitHub…" : status}</h2>
                  {state?.available && !state.blocked_reason && <p className="mb-0 mt-2 text-sm text-muted-foreground">{state.behind} new commit{state.behind === 1 ? "" : "s"} can be installed.</p>}
                  {!state?.available && !state?.error && state && <p className="mb-0 mt-2 text-sm text-muted-foreground">Your checkout matches {state.upstream}.</p>}
                </div>
                <Badge variant={state?.can_update ? "success" : state?.blocked_reason || state?.error ? "warning" : "secondary"}>{status}</Badge>
              </div>

              {state?.blocked_reason && <div className="mt-5 rounded-lg border border-amber-400/25 bg-amber-400/5 px-4 py-3 text-sm text-amber-100">{state.blocked_reason}</div>}
              {state?.error && <div className="mt-5 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">{state.error}</div>}

              <div className="mt-6 grid gap-4 border-t pt-5 sm:grid-cols-2">
                <div><p className="m-0 text-xs text-muted-foreground">Installed</p><p className="mb-0 mt-1 font-medium">Version {state?.version || "—"} · {state?.local_short || "—"}</p></div>
                <div><p className="m-0 text-xs text-muted-foreground">Latest on GitHub</p><p className="mb-0 mt-1 font-medium">{state?.remote_short || "—"}</p></div>
                <div><p className="m-0 text-xs text-muted-foreground">Branch</p><p className="mb-0 mt-1 break-all font-medium">{state?.branch || "—"} → {state?.upstream || "—"}</p></div>
                <div><p className="m-0 text-xs text-muted-foreground">Latest commit</p><p className="mb-0 mt-1 font-medium">{state?.latest_subject || "—"}</p><p className="mb-0 mt-1 text-xs text-muted-foreground">{shortDate(state?.latest_date)}</p></div>
              </div>

              <div className="mt-6 flex flex-wrap items-center gap-3">
                <Button onClick={() => void install()} disabled={!state?.can_update || checking || installing}>{installing ? "Installing…" : state?.supervised_restart ? "Update and restart" : "Install update"}</Button>
                {!state?.supervised_restart && state?.can_update && <p className="m-0 text-xs text-muted-foreground">Started outside START.bat; a manual restart will be required.</p>}
                {state?.restart_required && !state.restart_scheduled && <p className="m-0 text-sm text-amber-200">Update installed. Close and start Reference Desk again.</p>}
              </div>
            </CardContent>
          </Card>

          {!!state?.changed_paths?.length && <Card className="bg-card/75"><CardContent className="p-6"><h3 className="m-0 text-base font-semibold">Local source changes</h3><p className="mb-0 mt-2 text-sm text-muted-foreground">These tracked files must be committed or restored before the updater can continue.</p><div className="mt-4 max-h-64 overflow-auto rounded-lg border bg-background/50 p-3 font-mono text-xs leading-6">{state.changed_paths.map((path) => <div key={path}>{path}</div>)}</div></CardContent></Card>}
        </section>

        <aside className="space-y-5">
          <Card className="bg-card/70"><CardContent className="p-5"><h3 className="m-0 text-base font-semibold">Repository</h3><p className="mb-0 mt-3 break-all text-sm text-muted-foreground">{state?.remote_url || "Not detected"}</p></CardContent></Card>
          <Card className="bg-card/70"><CardContent className="p-5"><h3 className="m-0 text-base font-semibold">What is protected</h3><p className="mb-0 mt-3 text-sm leading-6 text-muted-foreground">Updates are fast-forward only. They never reset the checkout or replace local source edits. PDFs, indexes, settings, backups, benchmarks, and workspace data are outside the Git update.</p></CardContent></Card>
          <Card className="bg-card/70"><CardContent className="p-5"><h3 className="m-0 text-base font-semibold">If an update is blocked</h3><p className="mb-0 mt-3 text-sm leading-6 text-muted-foreground">Use Git to commit or restore the listed source files. The updater will not make that decision for you.</p></CardContent></Card>
        </aside>
      </div>
    </div>
  )
}
