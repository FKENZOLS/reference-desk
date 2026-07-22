import { Archive, ArchiveRestore, Download, FilePlus2, FileText, FolderOpen, HardDrive, LoaderCircle, Pause, Pencil, Play, RefreshCw, RotateCcw, Search, ShieldAlert, ShieldCheck, Trash2, Upload } from "lucide-react"
import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react"
import { toast } from "sonner"
import { EmptyState } from "@/components/empty-state"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Modal } from "@/components/ui/modal"
import { Progress } from "@/components/ui/progress"
import { Switch } from "@/components/ui/switch"
import { api, jsonRequest } from "@/lib/api"
import { formatBytes, formatDate } from "@/lib/utils"
import type { CorpusHealth, DocumentState, ManagedDocument } from "@/types"

const emptyState: DocumentState = {
  documents: [], trash: [], quarantine: [], revisions: [], backups: [], counts: { documents: 0, indexed: 0, pending: 0, trash: 0, quarantine: 0, revisions: 0 }, job: {}, app_instance_id: "",
}

export function DocumentsPage() {
  const [state, setState] = useState<DocumentState>(emptyState)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [filter, setFilter] = useState("")
  const [folder, setFolder] = useState("")
  const [replace, setReplace] = useState(false)
  const [moveItem, setMoveItem] = useState<ManagedDocument | null>(null)
  const [target, setTarget] = useState("")
  const [dragging, setDragging] = useState(false)
  const [backupBusy, setBackupBusy] = useState(false)
  const [optimizeBusy, setOptimizeBusy] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  async function refresh(silent = false) {
    try {
      const value = await api<DocumentState>(`/documents/api/list?t=${Date.now()}`)
      setState(value)
    } catch (error) {
      if (!silent) toast.error(error instanceof Error ? error.message : "Could not load library")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void refresh() }, [])
  useEffect(() => {
    if (!state.job.running) return
    const timer = window.setInterval(() => void refresh(true), 1200)
    return () => window.clearInterval(timer)
  }, [state.job.running])

  const filtered = useMemo(() => {
    const needle = filter.toLowerCase().trim()
    return state.documents.filter((document) => !needle || `${document.source_id} ${document.status}`.toLowerCase().includes(needle))
  }, [filter, state.documents])

  async function uploadFiles(files: File[]) {
    const pdfs = files.filter((file) => file.name.toLowerCase().endsWith(".pdf"))
    if (!pdfs.length) return toast.error("Choose one or more PDF files")
    setUploading(true)
    try {
      for (const file of pdfs) {
        const cleanFolder = folder.trim().replaceAll("\\", "/").replace(/^\/+|\/+$/g, "")
        const path = cleanFolder ? `${cleanFolder}/${file.name}` : file.name
        await api(`/documents/api/upload?path=${encodeURIComponent(path)}&replace=${replace}`, {
          method: "POST", headers: { "Content-Type": "application/pdf" }, body: file,
        })
      }
      toast.success(`${pdfs.length} PDF${pdfs.length === 1 ? "" : "s"} added. Apply changes when ready.`)
      if (inputRef.current) inputRef.current.value = ""
      await refresh(true)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Upload failed")
    } finally {
      setUploading(false)
    }
  }

  async function sync(force: boolean) {
    if (force && !window.confirm("Reindex every document? This can take a long time.")) return
    try {
      const job = await api<DocumentState["job"]>("/documents/api/sync", jsonRequest("POST", { force }))
      setState((current) => ({ ...current, job }))
      toast.success(force ? "Full reindex started" : "Index update started")
      await refresh(true)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not start indexing")
    }
  }

  async function moveDocument() {
    if (!moveItem || !target.trim() || target === moveItem.source_id) return
    try {
      await api(`/documents/api/item/${encodeURIComponent(moveItem.source_id)}`, jsonRequest("PATCH", { target: target.trim() }))
      setMoveItem(null)
      toast.success("Document moved")
      await refresh(true)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not move document")
    }
  }

  async function trashDocument(document: ManagedDocument) {
    if (!window.confirm(`Move ${document.name} to the recoverable trash?`)) return
    try {
      await api(`/documents/api/item/${encodeURIComponent(document.source_id)}`, { method: "DELETE" })
      toast.success("Document moved to trash")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not delete document") }
  }

  async function restore(trashId: string) {
    try {
      await api(`/documents/api/trash/${encodeURIComponent(trashId)}/restore`, { method: "POST" })
      toast.success("Document restored")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not restore document") }
  }

  async function purge(trashId: string) {
    if (!window.confirm("Permanently delete this PDF? This cannot be undone.")) return
    try {
      await api(`/documents/api/trash/${encodeURIComponent(trashId)}`, { method: "DELETE" })
      toast.success("Document permanently deleted")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not delete document") }
  }

  async function pauseQueue() {
    try {
      await api("/documents/api/queue/pause", { method: "POST" })
      toast.message("Pause requested. The current document will finish first.")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not pause ingestion") }
  }

  async function resumeQueue() {
    try {
      const job = await api<DocumentState["job"]>("/documents/api/queue/resume", { method: "POST" })
      setState((current) => ({ ...current, job }))
      toast.success("Ingestion resumed")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not resume ingestion") }
  }

  async function restoreQuarantine(quarantineId: string) {
    try {
      await api(`/documents/api/quarantine/${encodeURIComponent(quarantineId)}/restore`, { method: "POST" })
      toast.success("Document returned to the queue")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not restore document") }
  }

  async function deleteQuarantine(quarantineId: string) {
    if (!window.confirm("Permanently delete this quarantined PDF?")) return
    try {
      await api(`/documents/api/quarantine/${encodeURIComponent(quarantineId)}`, { method: "DELETE" })
      toast.success("Quarantined document deleted")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not delete document") }
  }

  async function restoreRevision(revisionId: string) {
    if (!window.confirm("Restore this earlier revision? The current version will be kept in revision history.")) return
    try {
      await api(`/documents/api/revisions/${encodeURIComponent(revisionId)}/restore`, { method: "POST" })
      toast.success("Revision restored. Apply changes to update the index.")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not restore revision") }
  }

  async function refreshHealth() {
    try {
      const health = await api<CorpusHealth>(`/documents/api/health?refresh=true&t=${Date.now()}`)
      setState((current) => ({ ...current, health }))
      toast.success("Corpus health refreshed")
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not refresh corpus health") }
  }

  async function createBackup() {
    setBackupBusy(true)
    try {
      await api("/documents/api/backups", jsonRequest("POST", {}))
      toast.success("Corpus backup created")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not create backup") }
    finally { setBackupBusy(false) }
  }

  async function optimizeStorage() {
    if (!window.confirm("Optimize the local index now? Search will pause, a recovery backup will be created, and active passages will be verified before old storage is removed.")) return
    setOptimizeBusy(true)
    try {
      const result = await api<{ reclaimed_bytes: number; chunks_verified: number }>("/documents/api/optimize", { method: "POST" })
      toast.success(`Storage optimized. Reclaimed ${formatBytes(result.reclaimed_bytes)} and verified ${result.chunks_verified.toLocaleString()} passages.`)
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not optimize storage") }
    finally { setOptimizeBusy(false) }
  }

  async function restoreBackup(backupId: string) {
    if (!window.confirm("Restore this snapshot? Current documents, indexes, quarantine, revisions, and workspace data will be replaced.")) return
    setBackupBusy(true)
    try {
      await api(`/documents/api/backups/${encodeURIComponent(backupId)}/restore`, { method: "POST" })
      toast.success("Corpus restored")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not restore backup") }
    finally { setBackupBusy(false) }
  }

  async function deleteBackup(backupId: string) {
    if (!window.confirm("Delete this backup archive?")) return
    try {
      await api(`/documents/api/backups/${encodeURIComponent(backupId)}`, { method: "DELETE" })
      toast.success("Backup deleted")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not delete backup") }
  }

  async function reloadModels() {
    try {
      const restart = await api<{ instance_id: string }>("/documents/api/restart", { method: "POST" })
      toast.message("Reloading search models…")
      for (let attempt = 0; attempt < 80; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 500))
        try {
          const current = await api<DocumentState>(`/documents/api/list?restart=${Date.now()}`)
          if (current.app_instance_id !== restart.instance_id) {
            setState(current)
            toast.success("Search models are ready")
            return
          }
        } catch { /* The app may briefly be unavailable. */ }
      }
      toast.error("The models are still reloading. Try again shortly.")
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not reload models") }
  }

  const jobState = String(state.job.state || "")
  const showJob = Boolean(state.job.running || state.queue?.remaining || ["complete", "complete_with_failures", "failed", "paused"].includes(jobState))
  const logLines = Array.isArray(state.job.log) ? state.job.log : []
  const queueItems = state.queue?.items || []
  const queueDone = queueItems.filter((item) => ["complete", "failed", "quarantined"].includes(item.status)).length
  const queueProgress = queueItems.length ? Math.round((queueDone / queueItems.length) * 100) : (jobState === "complete" ? 100 : 0)
  const health = state.health
  const healthVariant = health?.status === "healthy" ? "default" : health?.status === "critical" ? "destructive" : "secondary"

  return (
    <div>
      <PageHeader title="Documents" actions={<>
        <Button variant="outline" asChild><a href="/api/diagnostics/export"><Download /> Export diagnostics</a></Button>
        <Button variant="outline" onClick={() => void sync(true)} disabled={Boolean(state.job.running)}><RefreshCw /> Reindex all</Button>
        <Button onClick={() => void sync(false)} disabled={!state.counts.pending || Boolean(state.job.running)}>{state.job.running ? <LoaderCircle className="animate-spin" /> : <RotateCcw />} Apply changes{state.counts.pending ? ` (${state.counts.pending})` : ""}</Button>
      </>} />

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        {[['Documents', state.counts.documents], ['Indexed', state.counts.indexed], ['Pending', state.counts.pending], ['Trash', state.counts.trash]].map(([label, value]) => (
          <Card key={String(label)} className="bg-card/70"><CardContent className="p-4"><p className="m-0 text-xs text-muted-foreground">{label}</p><p className="mb-0 mt-2 text-2xl font-semibold tabular-nums">{value}</p></CardContent></Card>
        ))}
      </div>

      <Card className="mb-5 bg-card/75">
        <CardContent className="p-5">
          <div className="grid gap-4 xl:grid-cols-[1fr_250px_auto] xl:items-end">
            <div
              className={`flex min-h-24 cursor-pointer items-center justify-center rounded-xl border border-dashed px-5 text-center transition-colors ${dragging ? "border-primary bg-primary/8" : "bg-muted/20 hover:bg-muted/35"}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); void uploadFiles([...event.dataTransfer.files]) }}
            >
              <input ref={inputRef} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(event: ChangeEvent<HTMLInputElement>) => void uploadFiles([...(event.target.files || [])])} />
              <div><Upload className="mx-auto mb-2 size-5 text-primary" /><p className="m-0 font-medium">{uploading ? "Uploading…" : "Add PDF files"}</p><p className="mb-0 mt-1 text-xs text-muted-foreground">Click or drop files here</p></div>
            </div>
            <div><Label htmlFor="destination-folder">Destination folder</Label><Input id="destination-folder" value={folder} onChange={(event) => setFolder(event.target.value)} placeholder="standards/monorail" /></div>
            <div className="pb-2"><Switch checked={replace} onCheckedChange={setReplace} label="Replace matching files" /></div>
          </div>
        </CardContent>
      </Card>

      {showJob && (
        <Card className={`mb-5 ${jobState === "failed" ? "border-destructive/45" : "border-primary/30"}`}>
          <CardContent className="p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div><p className="m-0 font-semibold">{jobState === "complete" ? "Index updated" : jobState === "complete_with_failures" ? "Index updated with quarantined files" : jobState === "failed" ? "Index update stopped" : jobState === "paused" ? "Ingestion paused" : "Ingestion queue"}</p>{queueItems.length > 0 && <p className="mb-0 mt-1 text-xs text-muted-foreground">{queueDone} of {queueItems.length} finished</p>}</div>
              <div className="flex gap-2">
                {state.job.running && <Button size="sm" variant="outline" onClick={pauseQueue} disabled={Boolean(state.job.pause_requested)}>{state.job.pause_requested ? <LoaderCircle className="animate-spin" /> : <Pause />} {state.job.pause_requested ? "Pausing" : "Pause"}</Button>}
                {!state.job.running && state.queue?.remaining ? <Button size="sm" onClick={resumeQueue}><Play /> Resume</Button> : null}
                <Badge variant="secondary">{jobState || "queued"}</Badge>
              </div>
            </div>
            <Progress value={queueProgress} className="my-3" />
            <p className="m-0 text-sm text-muted-foreground">{String(state.job.message || "")}</p>
            {!!queueItems.length && <details className="mt-3"><summary className="cursor-pointer text-xs text-muted-foreground">Queue details</summary><div className="mt-2 space-y-2">{queueItems.map((item) => <div key={item.id} className="flex items-center justify-between gap-3 rounded-lg bg-background/50 px-3 py-2 text-xs"><span className="truncate">{item.action === "delete" ? "Remove index: " : ""}{item.source_id}</span><Badge variant={item.status === "quarantined" || item.status === "failed" ? "destructive" : "secondary"}>{item.status}</Badge></div>)}</div></details>}
            {!!logLines.length && <details className="mt-3"><summary className="cursor-pointer text-xs text-muted-foreground">Technical log</summary><pre className="scrollbar-thin mt-2 max-h-36 overflow-auto rounded-lg bg-background p-3 text-[11px] leading-5 text-muted-foreground">{logLines.join("\n")}</pre></details>}
            {["complete", "complete_with_failures"].includes(jobState) && <div className="mt-4"><Button onClick={reloadModels}>Reload search models</Button></div>}
          </CardContent>
        </Card>
      )}

      <div className="mb-5 grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Card className="bg-card/75">
          <CardContent className="p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3"><div className="grid size-9 place-items-center rounded-lg bg-primary/10 text-primary">{health?.status === "healthy" ? <ShieldCheck /> : <ShieldAlert />}</div><div><p className="m-0 font-semibold">Corpus health</p><p className="mb-0 mt-1 text-xs text-muted-foreground">{health ? `${state.hardware?.backend === "rocm" ? "AMD ROCm" : state.hardware?.backend === "cuda" ? "NVIDIA CUDA" : "CPU"} · ${health.pages.toLocaleString()} pages · ${health.chunks.toLocaleString()} passages` : "Calculating"}</p></div></div>
              <div className="flex items-center gap-2"><Badge variant={healthVariant}>{health?.status || "loading"}</Badge><Button size="sm" variant="outline" onClick={optimizeStorage} disabled={optimizeBusy || Boolean(state.job.running) || Boolean(state.counts.pending)}>{optimizeBusy ? <LoaderCircle className="animate-spin" /> : null} Optimize storage</Button><Button size="icon" variant="ghost" aria-label="Refresh corpus health" onClick={refreshHealth}><RefreshCw /></Button></div>
            </div>
            {health && <>
              <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
                {[['PDFs', health.storage.documents], ['Index', health.storage.index], ['Workspace', health.storage.workspace], ['Active total', health.storage.active]].map(([label, value]) => <div key={String(label)} className="rounded-lg border bg-background/45 p-3"><p className="m-0 text-xs text-muted-foreground">{label}</p><p className="mb-0 mt-1 font-semibold tabular-nums">{formatBytes(Number(value))}</p></div>)}
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
                {[['Quarantine', health.storage.quarantine], ['Revisions', health.storage.revisions], ['Backups', health.storage.backups], ['Total', health.storage.total]].map(([label, value]) => <div key={String(label)} className="flex items-center justify-between gap-2 rounded-lg bg-muted/20 px-3 py-2 text-xs"><span className="text-muted-foreground">{label}</span><span className="tabular-nums">{formatBytes(Number(value))}</span></div>)}
              </div>
              {Boolean(health.storage.reclaimable) && <p className="mb-0 mt-3 text-xs text-muted-foreground">At least {formatBytes(Number(health.storage.reclaimable))} can be reclaimed. Rebuilding the vector index may recover additional reserved capacity.</p>}
              {!!health.issues.length && <div className="mt-4 space-y-2">{health.issues.map((issue) => <div key={issue.label} className="flex items-center justify-between gap-3 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-sm"><span>{issue.label}</span><Badge variant="secondary">{issue.count}</Badge></div>)}</div>}
            </>}
          </CardContent>
        </Card>

        <Card className="bg-card/75">
          <CardContent className="p-5">
            <div className="flex items-center justify-between gap-3"><div className="flex items-center gap-2"><Archive className="size-4 text-primary" /><p className="m-0 font-semibold">Backups</p></div><Button size="sm" variant="outline" onClick={createBackup} disabled={backupBusy || Boolean(state.job.running)}>{backupBusy ? <LoaderCircle className="animate-spin" /> : <HardDrive />} Create</Button></div>
            <div className="mt-4 space-y-2">
              {(state.backups || []).length ? (state.backups || []).map((backup) => <div key={backup.backup_id} className="rounded-lg border bg-background/45 p-3"><div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="m-0 truncate text-sm font-medium">{backup.label || formatDate(backup.created_at)}</p><p className="mb-0 mt-1 text-xs text-muted-foreground">{formatBytes(backup.size)}</p></div><div className="flex gap-1"><Button size="icon" variant="ghost" aria-label="Restore backup" onClick={() => void restoreBackup(backup.backup_id)} disabled={backupBusy}><ArchiveRestore /></Button><Button size="icon" variant="ghost" className="text-destructive" aria-label="Delete backup" onClick={() => void deleteBackup(backup.backup_id)} disabled={backupBusy}><Trash2 /></Button></div></div></div>) : <div className="rounded-lg border border-dashed p-4 text-center text-xs text-muted-foreground">No backups yet</div>}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-card/75">
        <CardContent className="p-0">
          <div className="flex flex-wrap gap-3 border-b p-4">
            <div className="relative min-w-60 flex-1"><Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter documents" className="pl-9" /></div>
            <Button variant="outline" onClick={() => void refresh()}><RefreshCw /> Refresh</Button>
          </div>
          {loading ? <div className="p-10 text-center text-muted-foreground">Loading library…</div> : filtered.length ? (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left">
                <thead><tr className="border-b text-[11px] uppercase tracking-wide text-muted-foreground"><th className="px-4 py-3 font-medium">Document</th><th className="hidden px-4 py-3 font-medium md:table-cell">Pages</th><th className="hidden px-4 py-3 font-medium lg:table-cell">Size</th><th className="hidden px-4 py-3 font-medium lg:table-cell">Modified</th><th className="px-4 py-3 font-medium">Status</th><th className="px-4 py-3 font-medium">Actions</th></tr></thead>
                <tbody>{filtered.map((document) => <tr key={document.source_id} className="border-b last:border-0 hover:bg-muted/20">
                  <td className="max-w-xl px-4 py-3"><div className="flex items-center gap-3"><div className="grid size-8 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary"><FileText className="size-4" /></div><div className="min-w-0"><div className="flex items-center gap-2"><p className="m-0 truncate font-medium">{document.name}</p>{Boolean(document.revision_count) && <Badge variant="secondary">{document.revision_count} revision{document.revision_count === 1 ? "" : "s"}</Badge>}</div><p className="m-0 truncate text-xs text-muted-foreground">{document.source_id}</p></div></div></td>
                  <td className="hidden px-4 py-3 tabular-nums md:table-cell">{document.pages ?? "—"}</td>
                  <td className="hidden px-4 py-3 text-muted-foreground lg:table-cell">{formatBytes(document.size)}</td>
                  <td className="hidden px-4 py-3 text-muted-foreground lg:table-cell">{formatDate(document.modified_at)}</td>
                  <td className="px-4 py-3"><Badge variant={document.status === "indexed" ? "default" : document.status === "failed" ? "destructive" : "secondary"}>{document.status}</Badge>{document.state_updated_at && <p className="mb-0 mt-1 text-[10px] text-muted-foreground">{formatDate(document.state_updated_at)}</p>}{Boolean(document.state_history?.length) && <details className="mt-1"><summary className="cursor-pointer text-[10px] text-muted-foreground">Lifecycle</summary><div className="mt-1 w-64 space-y-1 rounded-md border bg-popover p-2">{document.state_history?.slice().reverse().slice(0, 6).map((event, index) => <div key={`${event.at}-${index}`} className="text-[10px]"><span className="font-medium">{event.from ? `${event.from} → ` : ""}{event.to}</span><span className="text-muted-foreground"> · {formatDate(event.at)}</span>{event.reason && <p className="m-0 text-muted-foreground">{event.reason}</p>}</div>)}</div></details>}</td>
                  <td className="px-4 py-3"><div className="flex gap-1"><Button size="icon" variant="ghost" aria-label="Move or rename" onClick={() => { setMoveItem(document); setTarget(document.source_id) }}><Pencil /></Button><Button size="icon" variant="ghost" className="text-destructive" aria-label="Delete" onClick={() => void trashDocument(document)}><Trash2 /></Button></div></td>
                </tr>)}</tbody>
              </table>
            </div>
          ) : <div className="p-4"><EmptyState icon={filter ? Search : FilePlus2} title={filter ? "No documents match this filter" : "Your library is empty"} /></div>}
        </CardContent>
      </Card>

      {!!state.quarantine?.length && <details open className="mt-4 rounded-xl border border-amber-400/20 bg-card/60 p-4"><summary className="cursor-pointer font-medium text-amber-100">Quarantine ({state.quarantine.length})</summary><div className="mt-3 space-y-2">{state.quarantine.map((document) => <div key={document.quarantine_id} className="flex flex-wrap items-start justify-between gap-3 rounded-lg border bg-background/50 p-3"><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><ShieldAlert className="size-4 text-amber-300" /><p className="m-0 truncate font-medium">{document.source_id}</p></div><p className="mb-0 mt-1 line-clamp-2 text-xs text-muted-foreground">{document.error}</p><p className="mb-0 mt-1 text-xs text-muted-foreground">{formatBytes(document.size)} · {formatDate(document.quarantined_at)}</p></div><div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void restoreQuarantine(document.quarantine_id)}><RotateCcw /> Retry</Button><Button size="sm" variant="destructive" onClick={() => void deleteQuarantine(document.quarantine_id)}><Trash2 /> Delete</Button></div></div>)}</div></details>}

      {!!state.revisions?.length && <details className="mt-4 rounded-xl border bg-card/60 p-4"><summary className="cursor-pointer font-medium">Revision history ({state.revisions.length})</summary><div className="mt-3 space-y-2">{state.revisions.map((revision) => <div key={revision.revision_id} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-background/50 p-3"><div><p className="m-0 font-medium">{revision.source_id}</p><p className="mb-0 mt-1 text-xs text-muted-foreground">{formatDate(revision.created_at)} · {formatBytes(revision.size)}</p></div><Button size="sm" variant="outline" onClick={() => void restoreRevision(revision.revision_id)}><ArchiveRestore /> Restore revision</Button></div>)}</div></details>}

      {!!state.trash.length && <details className="mt-4 rounded-xl border bg-card/60 p-4"><summary className="cursor-pointer font-medium">Trash ({state.trash.length})</summary><div className="mt-3 space-y-2">{state.trash.map((document) => <div key={document.trash_id} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-background/50 p-3"><div><p className="m-0 font-medium">{document.source_id}</p><p className="m-0 text-xs text-muted-foreground">{formatBytes(document.size)}</p></div><div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => void restore(document.trash_id)}><ArchiveRestore /> Restore</Button><Button size="sm" variant="destructive" onClick={() => void purge(document.trash_id)}><Trash2 /> Delete forever</Button></div></div>)}</div></details>}

      <Modal open={Boolean(moveItem)} title="Move or rename document" onClose={() => setMoveItem(null)}>
        <Label htmlFor="target-path">Path inside the library</Label><Input id="target-path" value={target} onChange={(event) => setTarget(event.target.value)} autoFocus />
        <div className="mt-5 flex justify-end gap-2"><Button variant="ghost" onClick={() => setMoveItem(null)}>Cancel</Button><Button onClick={moveDocument}><FolderOpen /> Save path</Button></div>
      </Modal>
    </div>
  )
}
