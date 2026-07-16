import { Activity, CheckCircle2, Download, Gauge, RefreshCw, ShieldCheck, Trash2, XCircle } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { EmptyState } from "@/components/empty-state"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Switch } from "@/components/ui/switch"
import { api, jsonRequest } from "@/lib/api"
import { formatDate } from "@/lib/utils"
import type { Calibration, Feedback, QualityState } from "@/types"

const emptyState: QualityState = {
  summary: {
    total: 0,
    counts: { relevant: 0, wrong_passage: 0, wrong_document: 0, no_relevant_result: 0 },
    benchmark_cases: 0,
    answerable_cases: 0,
    unanswerable_cases: 0,
    calibration: { threshold: null, positive_count: 0, negative_count: 0, positive_recall: null, specificity: null, balanced_accuracy: null, ready: false, enabled: true, active: false, minimum_positive: 20, minimum_negative: 20, updated_at: "" },
  },
  feedback: [],
}

const judgmentLabels: Record<Feedback["judgment"], string> = {
  relevant: "Relevant",
  wrong_passage: "Wrong passage",
  wrong_document: "Wrong document",
  no_relevant_result: "No relevant result",
}

const judgmentVariants: Record<Feedback["judgment"], "success" | "warning" | "destructive" | "outline"> = {
  relevant: "success",
  wrong_passage: "warning",
  wrong_document: "destructive",
  no_relevant_result: "outline",
}

const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`

export function QualityPage() {
  const [state, setState] = useState<QualityState>(emptyState)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<"all" | Feedback["judgment"]>("all")

  async function refresh() {
    try { setState(await api<QualityState>("/quality/api/state")) }
    catch (error) { toast.error(error instanceof Error ? error.message : "Could not load quality data") }
    finally { setLoading(false) }
  }
  useEffect(() => { void refresh() }, [])

  const calibration = state.summary.calibration
  const feedback = useMemo(() => state.feedback.filter((item) => filter === "all" || item.judgment === filter), [filter, state.feedback])
  const positiveProgress = 100 * Math.min(1, calibration.positive_count / Math.max(1, calibration.minimum_positive))
  const negativeProgress = 100 * Math.min(1, calibration.negative_count / Math.max(1, calibration.minimum_negative))

  async function calibrate() {
    try {
      const value = await api<Calibration>("/quality/api/calibrate", { method: "POST" })
      setState((current) => ({ ...current, summary: { ...current.summary, calibration: value } }))
      toast.success(value.ready ? "Relevance gate recalibrated" : "Calibration updated; more labels are still needed")
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not calibrate") }
  }

  async function toggleGate(enabled: boolean) {
    try {
      const value = await api<Calibration>("/quality/api/calibration", jsonRequest("PATCH", { enabled }))
      setState((current) => ({ ...current, summary: { ...current.summary, calibration: value } }))
      toast.success(enabled ? "Relevance gate enabled" : "Relevance gate disabled")
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not change gate") }
  }

  async function removeFeedback(id: number) {
    if (!window.confirm("Remove this quality label?")) return
    try {
      await api(`/quality/api/feedback/${id}`, { method: "DELETE" })
      toast.success("Label removed")
      await refresh()
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not remove label") }
  }

  return (
    <div>
      <PageHeader title="Reference quality" actions={<>
        <Button variant="outline" asChild><a href="/quality/export/benchmark"><Download /> Export benchmark</a></Button>
        <Button onClick={calibrate}><RefreshCw /> Recalibrate</Button>
      </>} />

      <div className="mb-5 grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Card className="bg-card/70"><CardContent className="p-4"><p className="m-0 text-xs text-muted-foreground">Labels</p><p className="mb-0 mt-2 text-2xl font-semibold">{state.summary.total}</p></CardContent></Card>
        <Card className="bg-card/70"><CardContent className="p-4"><p className="m-0 text-xs text-muted-foreground">Benchmark cases</p><p className="mb-0 mt-2 text-2xl font-semibold">{state.summary.benchmark_cases}</p></CardContent></Card>
        <Card className="bg-card/70"><CardContent className="p-4"><p className="m-0 text-xs text-muted-foreground">Relevant</p><p className="mb-0 mt-2 text-2xl font-semibold text-emerald-300">{state.summary.counts.relevant || 0}</p></CardContent></Card>
        <Card className="bg-card/70"><CardContent className="p-4"><p className="m-0 text-xs text-muted-foreground">Rejected</p><p className="mb-0 mt-2 text-2xl font-semibold text-amber-300">{state.summary.total - (state.summary.counts.relevant || 0)}</p></CardContent></Card>
      </div>

      <Card className="mb-5 bg-card/75"><CardContent className="p-5 md:p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex gap-3"><div className="grid size-10 place-items-center rounded-xl bg-primary/10 text-primary"><ShieldCheck className="size-5" /></div><div><div className="flex items-center gap-2"><p className="m-0 font-semibold">Calibrated relevance gate</p><Badge variant={calibration.active ? "success" : calibration.ready ? "warning" : "outline"}>{calibration.active ? "Active" : calibration.ready ? "Paused" : "Collecting labels"}</Badge></div><p className="mb-0 mt-1 text-sm text-muted-foreground">{calibration.ready ? `Threshold ${calibration.threshold?.toFixed(3)}` : "Activates after enough positive and negative examples."}</p></div></div>
          <Switch checked={calibration.enabled} disabled={!calibration.ready} onCheckedChange={toggleGate} label="Use gate in search" />
        </div>
        <div className="mt-6 grid gap-5 md:grid-cols-2">
          <div><div className="mb-2 flex justify-between text-xs"><span>Relevant examples</span><span className="text-muted-foreground">{calibration.positive_count} / {calibration.minimum_positive}</span></div><Progress value={positiveProgress} /></div>
          <div><div className="mb-2 flex justify-between text-xs"><span>Negative examples</span><span className="text-muted-foreground">{calibration.negative_count} / {calibration.minimum_negative}</span></div><Progress value={negativeProgress} /></div>
        </div>
        {calibration.ready && <div className="mt-6 grid grid-cols-3 gap-3 border-t pt-5 text-center"><div><p className="m-0 text-lg font-semibold">{percent(calibration.positive_recall)}</p><p className="m-0 text-xs text-muted-foreground">Recall</p></div><div><p className="m-0 text-lg font-semibold">{percent(calibration.specificity)}</p><p className="m-0 text-xs text-muted-foreground">Specificity</p></div><div><p className="m-0 text-lg font-semibold">{percent(calibration.balanced_accuracy)}</p><p className="m-0 text-xs text-muted-foreground">Balanced accuracy</p></div></div>}
      </CardContent></Card>

      <Card className="bg-card/75"><CardContent className="p-0">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b p-4">
          <div className="flex items-center gap-2 font-semibold"><Activity className="size-4 text-primary" /> Quality labels</div>
          <div className="flex flex-wrap gap-1">{(["all", "relevant", "wrong_passage", "wrong_document", "no_relevant_result"] as const).map((value) => <Button key={value} size="sm" variant={filter === value ? "secondary" : "ghost"} onClick={() => setFilter(value)}>{value === "all" ? "All" : judgmentLabels[value]}</Button>)}</div>
        </div>
        {loading ? <div className="p-10 text-center text-muted-foreground">Loading labels…</div> : feedback.length ? (
          <div className="overflow-x-auto"><table className="w-full border-collapse text-left"><thead><tr className="border-b text-[11px] uppercase tracking-wide text-muted-foreground"><th className="px-4 py-3 font-medium">Query</th><th className="px-4 py-3 font-medium">Judgment</th><th className="hidden px-4 py-3 font-medium md:table-cell">Source</th><th className="hidden px-4 py-3 font-medium lg:table-cell">Score</th><th className="hidden px-4 py-3 font-medium xl:table-cell">Updated</th><th className="px-4 py-3"></th></tr></thead><tbody>{feedback.map((item) => <tr key={item.id} className="border-b last:border-0 hover:bg-muted/20"><td className="max-w-md px-4 py-3"><p className="m-0 line-clamp-2 font-medium">{item.query}</p>{item.section && <p className="m-0 truncate text-xs text-muted-foreground">{item.section}</p>}</td><td className="px-4 py-3"><Badge variant={judgmentVariants[item.judgment]}>{item.judgment === "relevant" ? <CheckCircle2 className="mr-1 size-3" /> : <XCircle className="mr-1 size-3" />}{judgmentLabels[item.judgment]}</Badge></td><td className="hidden max-w-xs px-4 py-3 md:table-cell"><p className="m-0 truncate text-sm">{item.document_title || item.source_id || "—"}</p>{item.page_start && <p className="m-0 text-xs text-muted-foreground">page {item.page_start}</p>}</td><td className="hidden px-4 py-3 tabular-nums text-muted-foreground lg:table-cell">{item.rerank_logit == null ? "—" : item.rerank_logit.toFixed(3)}</td><td className="hidden px-4 py-3 text-xs text-muted-foreground xl:table-cell">{formatDate(item.updated_at || item.created_at)}</td><td className="px-4 py-3"><Button size="icon" variant="ghost" className="text-destructive" onClick={() => void removeFeedback(item.id)} aria-label="Remove label"><Trash2 /></Button></td></tr>)}</tbody></table></div>
        ) : <div className="p-4"><EmptyState icon={Gauge} title="No quality labels yet" detail="Mark search results as relevant or incorrect to create a local retrieval benchmark." /></div>}
      </CardContent></Card>
    </div>
  )
}
