import { CheckCircle2, FlaskConical, LoaderCircle, Play, Star, Upload } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { EmptyState } from "@/components/empty-state"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { api, jsonRequest } from "@/lib/api"
import { formatDate } from "@/lib/utils"
import type { ExperimentState, ExperimentSummary, QualityExperiment } from "@/types"

const emptyState: ExperimentState = { benchmarks: [], experiments: [], production: null, running: false }
const percent = (value: number | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`
const seconds = (value: number | undefined) => value == null ? "—" : `${value.toFixed(2)}s`

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg border bg-background/45 p-3"><p className="m-0 text-[11px] text-muted-foreground">{label}</p><p className="mb-0 mt-1 font-semibold tabular-nums">{value}</p></div>
}

function ModelResults({ model, summary }: { model: string; summary: ExperimentSummary }) {
  return <div className="min-w-0 rounded-xl border bg-background/35 p-4">
    <div className="mb-3 flex items-center justify-between"><p className="m-0 font-semibold">{model.toUpperCase()}</p><Badge variant="outline">{summary.cases} cases</Badge></div>
    <div className="grid grid-cols-2 gap-2 xl:grid-cols-3">
      <Metric label="nDCG@5" value={percent(summary.ndcg_at_5)} />
      <Metric label="MRR@5" value={percent(summary.mrr_at_5)} />
      <Metric label="Retrieval recall" value={percent(summary.retrieval_recall)} />
      <Metric label="Selected recall" value={percent(summary.selected_recall)} />
      <Metric label="Hard-negative hits" value={percent(summary.hard_negative_hit_rate)} />
      <Metric label="Latency p50" value={seconds(summary.latency_p50_seconds)} />
      <Metric label="Latency p95" value={seconds(summary.latency_p95_seconds)} />
    </div>
    {summary.confidence_intervals?.ndcg_at_5 && <p className="mb-0 mt-3 text-xs text-muted-foreground">95% nDCG interval: {percent(summary.confidence_intervals.ndcg_at_5.lower)}–{percent(summary.confidence_intervals.ndcg_at_5.upper)}</p>}
  </div>
}

export function ExperimentsPage() {
  const [state, setState] = useState<ExperimentState>(emptyState)
  const [loading, setLoading] = useState(true)
  const [name, setName] = useState("")
  const [benchmarkKey, setBenchmarkKey] = useState("feedback:gte")
  const [reranker, setReranker] = useState<"gte" | "bge" | "both">("both")
  const [candidateCount, setCandidateCount] = useState(20)
  const [rerankWeight, setRerankWeight] = useState(0.6)
  const [passageMode, setPassageMode] = useState("metadata-child")
  const [split, setSplit] = useState<"all" | "calibration" | "test">("calibration")
  const [baselineId, setBaselineId] = useState("")
  const [maxNdcgDrop, setMaxNdcgDrop] = useState(0.03)

  async function refresh(silent = false) {
    try {
      const value = await api<ExperimentState>("/quality/api/experiments")
      setState(value)
      if (!value.benchmarks.some((item) => item.key === benchmarkKey) && value.benchmarks[0]) setBenchmarkKey(value.benchmarks[0].key)
    } catch (error) {
      if (!silent) toast.error(error instanceof Error ? error.message : "Could not load experiments")
    } finally { setLoading(false) }
  }

  useEffect(() => { void refresh() }, [])
  useEffect(() => {
    if (!state.running) return
    const timer = window.setInterval(() => void refresh(true), 2000)
    return () => window.clearInterval(timer)
  }, [state.running])

  const selectedBenchmark = useMemo(() => state.benchmarks.find((item) => item.key === benchmarkKey), [benchmarkKey, state.benchmarks])

  async function importBenchmark(file: File | undefined) {
    if (!file) return
    try {
      const content = await file.text()
      await api("/quality/api/benchmarks", jsonRequest("POST", { name: file.name.replace(/\.jsonl?$/i, ""), content }))
      toast.success("Benchmark imported")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not import benchmark") }
  }

  async function runExperiment() {
    try {
      await api("/quality/api/experiments", jsonRequest("POST", {
        name: name.trim() || `${reranker.toUpperCase()} comparison`, benchmark_key: benchmarkKey,
        reranker, candidate_count: candidateCount, rerank_weight: rerankWeight, passage_mode: passageMode,
        split, baseline_experiment_id: baselineId ? Number(baselineId) : null, max_ndcg_drop: maxNdcgDrop,
      }))
      toast.success("Experiment started in the background")
      setName("")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not start experiment") }
  }

  async function makeProduction(experiment: QualityExperiment) {
    try {
      await api(`/quality/api/experiments/${experiment.id}/production`, { method: "POST" })
      toast.success("Production search now uses this configuration")
      await refresh(true)
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not activate configuration") }
  }

  return <div>
    <PageHeader title="Evaluation experiments" actions={<label className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-md border bg-background px-3 text-sm font-medium hover:bg-accent"><Upload className="size-4" /> Import JSONL<input type="file" accept=".jsonl,.json,application/x-ndjson" className="hidden" onChange={(event) => { void importBenchmark(event.target.files?.[0]); event.currentTarget.value = "" }} /></label>} />

    {state.production && <div className="mb-5 flex flex-wrap items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/5 px-4 py-3 text-sm"><CheckCircle2 className="size-4 text-emerald-300" /><span><strong>Production default:</strong> {state.production.name} · {state.production.config.reranker.toUpperCase()} · {state.production.config.candidate_count} candidates · weight {state.production.config.rerank_weight.toFixed(2)} · {state.production.config.passage_mode}</span></div>}

    <Card className="mb-6 bg-card/75"><CardContent className="p-5 md:p-6">
      <div className="mb-5"><h2 className="m-0 text-base font-semibold">New experiment</h2><p className="mb-0 mt-1 text-sm text-muted-foreground">Retrieval runs once; selected rerankers receive the identical candidate pool for a fair comparison.</p></div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div><Label htmlFor="experiment-name">Name</Label><Input id="experiment-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. wider pool test" /></div>
        <div><Label htmlFor="benchmark">Benchmark</Label><select id="benchmark" value={benchmarkKey} onChange={(event) => setBenchmarkKey(event.target.value)} className="h-10 w-full rounded-md border bg-background px-3 text-sm">{state.benchmarks.map((item) => <option key={item.key} value={item.key}>{item.name} · v{item.version} · {item.case_count} cases</option>)}</select></div>
        <div><Label htmlFor="candidate-count">Rerank candidates</Label><Input id="candidate-count" type="number" min={1} max={80} value={candidateCount} onChange={(event) => setCandidateCount(Number(event.target.value))} /></div>
        <div><Label htmlFor="rerank-weight">Reranker weight</Label><Input id="rerank-weight" type="number" min={0} max={1} step={0.05} value={rerankWeight} onChange={(event) => setRerankWeight(Number(event.target.value))} /></div>
        <div><Label>Models</Label><div className="grid h-10 grid-cols-3 rounded-lg border bg-background">{(["gte", "bge", "both"] as const).map((value) => <button key={value} type="button" onClick={() => setReranker(value)} className={reranker === value ? "rounded-md bg-primary px-2 text-xs font-medium text-primary-foreground" : "rounded-md px-2 text-xs text-muted-foreground hover:bg-accent"}>{value.toUpperCase()}</button>)}</div></div>
        <div><Label htmlFor="passage-mode">Passage input</Label><select id="passage-mode" value={passageMode} onChange={(event) => setPassageMode(event.target.value)} className="h-10 w-full rounded-md border bg-background px-3 text-sm"><option value="child">Child only</option><option value="metadata-child">Metadata + child</option><option value="metadata-parent">Metadata + parent</option><option value="metadata-child-parent">Metadata + child + parent</option></select></div>
        <div><Label htmlFor="benchmark-split">Data split</Label><select id="benchmark-split" value={split} onChange={(event) => setSplit(event.target.value as typeof split)} className="h-10 w-full rounded-md border bg-background px-3 text-sm"><option value="test">Held-out test</option><option value="calibration">Calibration</option><option value="all">All cases</option></select></div>
        <div><Label htmlFor="baseline">Regression baseline</Label><select id="baseline" value={baselineId} onChange={(event) => setBaselineId(event.target.value)} className="h-10 w-full rounded-md border bg-background px-3 text-sm"><option value="">No baseline</option>{state.experiments.filter((item) => item.status === "complete").map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></div>
        <div><Label htmlFor="max-drop">Maximum nDCG drop</Label><Input id="max-drop" type="number" min={0} max={1} step={0.01} value={maxNdcgDrop} onChange={(event) => setMaxNdcgDrop(Number(event.target.value))} /></div>
        <div className="flex items-end"><Button className="w-full" disabled={state.running || !selectedBenchmark || selectedBenchmark.case_count < 1} onClick={() => void runExperiment()}>{state.running ? <LoaderCircle className="animate-spin" /> : <Play />} {state.running ? "Experiment running…" : "Run and save experiment"}</Button></div>
      </div>
      {selectedBenchmark?.metadata && <p className="mb-0 mt-4 text-xs text-muted-foreground">Splits: {Object.entries(selectedBenchmark.metadata.splits ?? {}).map(([key, count]) => `${key} ${count}`).join(" · ") || "not labeled"} · Languages: {Object.keys(selectedBenchmark.metadata.languages ?? {}).join(", ") || "und"} · Hard negatives: {selectedBenchmark.metadata.hard_negative_count ?? 0}</p>}
    </CardContent></Card>

    <div className="mb-3 flex items-center gap-2"><FlaskConical className="size-4 text-primary" /><h2 className="m-0 text-base font-semibold">Saved experiments</h2></div>
    {loading ? <div className="p-10 text-center text-muted-foreground">Loading experiments…</div> : state.experiments.length ? <div className="space-y-4">{state.experiments.map((experiment) => <Card key={experiment.id} className="bg-card/70"><CardContent className="p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3"><div><div className="flex items-center gap-2"><h3 className="m-0 font-semibold">{experiment.name}</h3><Badge variant={experiment.status === "complete" ? "success" : experiment.status === "failed" ? "destructive" : "secondary"}>{experiment.status === "running" && <LoaderCircle className="mr-1 size-3 animate-spin" />}{experiment.status}</Badge>{experiment.production && <Badge variant="outline"><Star className="mr-1 size-3" /> Production</Badge>}{experiment.results.regression && <Badge variant={experiment.results.regression.passed ? "success" : "destructive"}>{experiment.results.regression.passed ? "Regression passed" : "Regression failed"}</Badge>}</div><p className="mb-0 mt-1 text-xs text-muted-foreground">{experiment.benchmark_name} · {experiment.config.split ?? "all"} · {experiment.config.candidate_count} candidates · weight {experiment.config.rerank_weight.toFixed(2)} · {experiment.config.passage_mode} · {formatDate(experiment.updated_at)}</p>{experiment.config.reranker === "both" && <p className="mb-0 mt-1 text-xs text-muted-foreground">Comparison runs cannot become production defaults; run the winning model separately.</p>}</div>{experiment.status === "complete" && experiment.config.reranker !== "both" && !experiment.production && experiment.results.regression?.passed !== false && <Button size="sm" variant="outline" onClick={() => void makeProduction(experiment)}><Star /> Use in production</Button>}</div>
      {experiment.error && <p className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">{experiment.error}</p>}
      {experiment.results.models && <div className="grid gap-3 lg:grid-cols-2">{Object.entries(experiment.results.models).map(([model, result]) => <ModelResults key={model} model={model} summary={result.summary} />)}</div>}
    </CardContent></Card>)}</div> : <EmptyState icon={FlaskConical} title="No experiments yet" detail="Choose a benchmark and run a model or configuration comparison." />}
  </div>
}
