import { type FormEvent, useEffect, useState } from "react"
import { toast } from "sonner"
import { EmptyState } from "@/components/empty-state"
import { PageHeader } from "@/components/page-header"
import { SearchResultCard } from "@/components/search-result-card"
import { SearchScopeSelect } from "@/components/search-scope-select"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { api, jsonRequest } from "@/lib/api"
import type { RerankerOption, RerankerRuntimeStatus, RerankersResponse, SearchResponse, SearchSource } from "@/types"

export function SearchPage() {
  const params = new URLSearchParams(window.location.search)
  const [query, setQuery] = useState(params.get("q") || "")
  const [source, setSource] = useState(params.get("source") || "")
  const [section, setSection] = useState(params.get("section") || "")
  const [contentType, setContentType] = useState(params.get("type") || "")
  const [date, setDate] = useState(params.get("date") || "")
  const [sources, setSources] = useState<SearchSource[]>([{ label: "All documents", value: "" }])
  const [rerankers, setRerankers] = useState<RerankerOption[]>([])
  const [rerankerStatus, setRerankerStatus] = useState<RerankerRuntimeStatus | null>(null)
  const [reranker, setReranker] = useState(params.get("reranker") || "")
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [showMore, setShowMore] = useState(false)
  const [withinResults, setWithinResults] = useState(false)
  const [showDebug, setShowDebug] = useState(false)

  useEffect(() => {
    api<{ sources: SearchSource[] }>("/api/sources")
      .then((value) => setSources(value.sources))
      .catch(() => undefined)
    api<RerankersResponse>("/api/rerankers")
      .then((value) => {
        setRerankers(value.options)
        setRerankerStatus(value.status)
        setReranker((current) => value.options.some((item) => item.value === current) ? current : value.default)
      })
      .catch(() => undefined)
    const statusTimer = window.setInterval(() => {
      api<RerankerRuntimeStatus>("/api/rerankers/status")
        .then(setRerankerStatus)
        .catch(() => undefined)
    }, 1500)
    return () => window.clearInterval(statusTimer)
  }, [])

  async function runSearch(event?: FormEvent) {
    event?.preventDefault()
    const cleanQuery = query.trim()
    if (!cleanQuery) return
    setLoading(true)
    if (rerankerStatus && rerankerStatus.choice !== reranker) {
      setRerankerStatus({ ...rerankerStatus, status: "loading", choice: reranker, message: `Loading ${reranker.toUpperCase()}…` })
    }
    setShowMore(false)
    const nextParams: Record<string, string> = { q: cleanQuery }
    if (source) nextParams.source = source
    if (section) nextParams.section = section
    if (contentType) nextParams.type = contentType
    if (date) nextParams.date = date
    if (reranker) nextParams.reranker = reranker
    window.history.replaceState({}, "", `/?${new URLSearchParams(nextParams)}`)
    try {
      const value = await api<SearchResponse>("/api/search", jsonRequest("POST", {
        query: cleanQuery,
        source_filter: source,
        section_filter: section,
        content_filter: contentType,
        date_filter: date,
        within_results: withinResults,
        previous_result_ids: response?.result_ids || [],
        reranker_choice: reranker,
      }))
      setResponse(value)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Search failed")
    } finally {
      setLoading(false)
    }
  }

  async function restartWorker() {
    try {
      const value = await api<RerankerRuntimeStatus>("/api/rerankers/restart", jsonRequest("POST", { reranker_choice: reranker }))
      setRerankerStatus(value)
      toast.success("Reranker worker restarted")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not restart reranker worker")
    }
  }

  return (
    <div>
      <PageHeader title="Search documents" />
      <form onSubmit={runSearch}>
        <Card className="bg-card/80">
          <CardContent className="p-4 md:p-5">
            <div className="grid gap-3 md:grid-cols-[minmax(220px,30%)_1fr_auto] md:items-end">
              <div>
                <Label htmlFor="search-source">Search scope</Label>
                <SearchScopeSelect id="search-source" value={source} options={sources} onChange={setSource} />
              </div>
              <div>
                <Label htmlFor="search-query">Query</Label>
                <Input id="search-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find a definition, requirement, table, or passage…" className="h-10" />
              </div>
              <Button type="submit" disabled={loading || !query.trim() || Boolean(rerankerStatus?.restart_required)} className="h-10 px-6">
                {loading ? "Searching…" : "Search"}
              </Button>
            </div>
            <details className="group mt-4 border-t pt-4">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground">
                Filters and search within results
              </summary>
              <div className="mt-4 grid gap-3 md:grid-cols-4">
                <div>
                  <Label>Reranker</Label>
                  <div className="grid h-10 grid-cols-2 rounded-lg border bg-background p-0" role="group" aria-label="Reranker">
                    {rerankers.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        aria-pressed={reranker === option.value}
                        title={`${option.description} ${option.model}`}
                        onClick={() => setReranker(option.value)}
                        className={reranker === option.value
                          ? "rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground"
                          : "rounded-md px-3 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  {rerankerStatus && (
                    <div className="mt-1.5 flex items-center gap-2">
                      <p className={`m-0 text-[11px] ${rerankerStatus.restart_required || rerankerStatus.status === "failed" ? "text-destructive" : "text-muted-foreground"}`}>
                        {rerankerStatus.status === "loading" && "Loading · "}
                        {rerankerStatus.message}{rerankerStatus.status === "ready" && rerankerStatus.device ? ` · ${rerankerStatus.device}` : ""}{rerankerStatus.worker_pid ? ` · worker ${rerankerStatus.worker_pid}` : ""}
                      </p>
                      {(rerankerStatus.status === "failed" || rerankerStatus.restart_required) && <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-[11px]" onClick={() => void restartWorker()}>Restart</Button>}
                    </div>
                  )}
                </div>
                <div><Label htmlFor="section-filter">Section</Label><Input id="section-filter" value={section} onChange={(event) => setSection(event.target.value)} placeholder="e.g. 5.1.3" /></div>
                <div><Label htmlFor="type-filter">Content type</Label><Input id="type-filter" value={contentType} onChange={(event) => setContentType(event.target.value)} placeholder="e.g. table or requirement" /></div>
                <div><Label htmlFor="date-filter">Document date</Label><Input id="date-filter" value={date} onChange={(event) => setDate(event.target.value)} placeholder="e.g. 2026" /></div>
                <div style={{ gridColumn: "1 / -1" }}><Switch checked={withinResults} disabled={!response?.result_ids.length} onCheckedChange={setWithinResults} label="Search only within the current result set" /></div>
                <div style={{ gridColumn: "1 / -1" }}><Switch checked={showDebug} onCheckedChange={setShowDebug} label="Show retrieval diagnostics" /></div>
              </div>
            </details>
          </CardContent>
        </Card>
      </form>

      {loading && <div className="mt-6 space-y-3"><p className="m-0 text-center text-sm text-muted-foreground">{rerankerStatus?.status === "loading" ? rerankerStatus.message : `Searching and reranking with ${rerankers.find((item) => item.value === reranker)?.label || reranker.toUpperCase()}…`}</p><Skeleton className="h-56 w-full" /><Skeleton className="h-56 w-full" /></div>}

      {!loading && !response && (
        <div className="mt-6"><EmptyState title="Search your reference library" detail="Results open directly at the cited page and highlighted source region." /></div>
      )}

      {!loading && response && (
        <section className="mt-7">
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <div>
              <h2 className="m-0 text-base font-semibold">{response.results.length ? `${response.results.length} best matches` : "No strong match"}</h2>
              <p className="mt-1 text-xs text-muted-foreground">{response.metrics.total_seconds.toFixed(2)} seconds · {response.metrics.considered_count} passages considered · {response.reranker.label} reranker</p>
              {showDebug && <><p className="mb-0 mt-1 text-[11px] text-muted-foreground">dense {response.metrics.dense_seconds.toFixed(3)}s · lexical {response.metrics.lexical_seconds.toFixed(3)}s · rerank {response.metrics.rerank_seconds.toFixed(3)}s · model load {response.metrics.model_load_seconds.toFixed(3)}s · {response.metrics.reranker_device}</p>{response.metrics.retrieval_plan && <p className="mb-0 mt-1 text-[11px] text-muted-foreground">plan {response.metrics.retrieval_plan.strategy} · dense {response.metrics.retrieval_plan.dense_candidates} · lexical {response.metrics.retrieval_plan.lexical_candidates} · rerank {response.metrics.retrieval_plan.rerank_candidates} · confidence {(response.metrics.retrieval_plan.fusion_confidence * 100).toFixed(0)}% · {response.metrics.retrieval_plan.signals.join(", ")}</p>}</>}
            </div>
          </div>
          {response.gate.no_strong_evidence && (
            <div className="mb-4 rounded-lg border border-amber-400/20 bg-amber-400/5 px-4 py-3 text-sm text-amber-100">No passage cleared the calibrated relevance threshold.</div>
          )}
          {response.results.length ? (
            <div className="space-y-4">{response.results.map((result) => <SearchResultCard key={result.chunk_id} result={result} showDebug={showDebug} />)}</div>
          ) : (
            <EmptyState title="No matching passages" detail="Try a broader query, remove a filter, or inspect the additional reranked passages." />
          )}
          {!!response.additional_results.length && (
            <div className="mt-5">
              <Button variant="secondary" onClick={() => setShowMore((value) => !value)}>{showMore ? "Hide" : "Show"} {response.additional_results.length} more reranked sources</Button>
              {showMore && <div className="mt-4 space-y-4">{response.additional_results.map((result) => <SearchResultCard key={result.chunk_id} result={result} showDebug={showDebug} />)}</div>}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
