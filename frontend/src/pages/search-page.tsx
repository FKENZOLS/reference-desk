import { ChevronDown, CircleAlert, Filter, LoaderCircle, Search, SlidersHorizontal } from "lucide-react"
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
import type { SearchResponse, SearchSource } from "@/types"

export function SearchPage() {
  const params = new URLSearchParams(window.location.search)
  const [query, setQuery] = useState(params.get("q") || "")
  const [source, setSource] = useState(params.get("source") || "")
  const [section, setSection] = useState(params.get("section") || "")
  const [contentType, setContentType] = useState(params.get("type") || "")
  const [date, setDate] = useState(params.get("date") || "")
  const [sources, setSources] = useState<SearchSource[]>([{ label: "All documents", value: "" }])
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [showMore, setShowMore] = useState(false)
  const [withinResults, setWithinResults] = useState(false)

  useEffect(() => {
    api<{ sources: SearchSource[] }>("/api/sources")
      .then((value) => setSources(value.sources))
      .catch(() => undefined)
  }, [])

  async function runSearch(event?: FormEvent) {
    event?.preventDefault()
    const cleanQuery = query.trim()
    if (!cleanQuery) return
    setLoading(true)
    setShowMore(false)
    const nextParams: Record<string, string> = { q: cleanQuery }
    if (source) nextParams.source = source
    if (section) nextParams.section = section
    if (contentType) nextParams.type = contentType
    if (date) nextParams.date = date
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
      }))
      setResponse(value)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Search failed")
    } finally {
      setLoading(false)
    }
  }

  async function noRelevantResult() {
    if (!response) return
    try {
      await api("/quality/api/feedback", jsonRequest("POST", { ...response.query_feedback, judgment: "no_relevant_result" }))
      toast.success("Search marked for review")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save feedback")
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
              <Button type="submit" disabled={loading || !query.trim()} className="h-10 px-6">
                {loading ? <LoaderCircle className="animate-spin" /> : <Search />} Search
              </Button>
            </div>
            <details className="group mt-4 border-t pt-4">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground">
                <SlidersHorizontal className="size-3.5" /> Filters and search within results <ChevronDown className="ml-auto size-3.5 transition-transform group-open:rotate-180" />
              </summary>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                <div><Label htmlFor="section-filter">Section</Label><Input id="section-filter" value={section} onChange={(event) => setSection(event.target.value)} placeholder="e.g. 5.1.3" /></div>
                <div><Label htmlFor="type-filter">Content type</Label><Input id="type-filter" value={contentType} onChange={(event) => setContentType(event.target.value)} placeholder="e.g. table or requirement" /></div>
                <div><Label htmlFor="date-filter">Document date</Label><Input id="date-filter" value={date} onChange={(event) => setDate(event.target.value)} placeholder="e.g. 2026" /></div>
                <div className="md:col-span-3"><Switch checked={withinResults} disabled={!response?.result_ids.length} onCheckedChange={setWithinResults} label="Search only within the current result set" /></div>
              </div>
            </details>
          </CardContent>
        </Card>
      </form>

      {loading && <div className="mt-6 space-y-3"><Skeleton className="h-56 w-full" /><Skeleton className="h-56 w-full" /></div>}

      {!loading && !response && (
        <div className="mt-6"><EmptyState icon={Search} title="Search your reference library" detail="Results open directly at the cited page and highlighted source region." /></div>
      )}

      {!loading && response && (
        <section className="mt-7">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="m-0 text-base font-semibold">{response.results.length ? `${response.results.length} best matches` : "No strong match"}</h2>
              <p className="mt-1 text-xs text-muted-foreground">{response.metrics.total_seconds.toFixed(2)} seconds · {response.metrics.considered_count} passages considered</p>
            </div>
            <Button size="sm" variant="ghost" onClick={noRelevantResult}><CircleAlert /> No relevant result</Button>
          </div>
          {response.gate.no_strong_evidence && (
            <div className="mb-4 flex items-center gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-4 py-3 text-sm text-amber-100"><CircleAlert className="size-4" /> No passage cleared the calibrated relevance threshold.</div>
          )}
          {response.results.length ? (
            <div className="space-y-4">{response.results.map((result) => <SearchResultCard key={result.chunk_id} result={result} />)}</div>
          ) : (
            <EmptyState icon={Filter} title="No matching passages" detail="Try a broader query, remove a filter, or inspect the additional reranked passages." />
          )}
          {!!response.additional_results.length && (
            <div className="mt-5">
              <Button variant="secondary" onClick={() => setShowMore((value) => !value)}>{showMore ? "Hide" : "Show"} {response.additional_results.length} more reranked sources</Button>
              {showMore && <div className="mt-4 space-y-4">{response.additional_results.map((result) => <SearchResultCard key={result.chunk_id} result={result} />)}</div>}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
