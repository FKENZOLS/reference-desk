import { useState } from "react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { api, jsonRequest } from "@/lib/api"
import type { SearchResult } from "@/types"

const copyText = async (value: string, message: string) => {
  await navigator.clipboard.writeText(value)
  toast.success(message)
}

export function SearchResultCard({ result, showDebug = false }: { result: SearchResult; showDebug?: boolean }) {
  const [saved, setSaved] = useState(false)
  const [judgment, setJudgment] = useState<"relevant" | "wrong_passage" | null>(null)
  const [feedbackId, setFeedbackId] = useState<number | null>(null)
  const [savingFeedback, setSavingFeedback] = useState(false)

  async function savePassage() {
    try {
      await api("/workspace/api/bookmarks", jsonRequest("POST", {
        chunk_id: result.chunk_id,
        source_id: result.source_id,
        document_title: result.document_title,
        page_start: result.page_start,
        page_end: result.page_end,
        section: result.section,
        content_type: result.content_type,
        document_date: result.document_date,
        excerpt: result.excerpt,
        citation_label: result.citation_label,
        citation_url: result.citation_url,
        query: result.feedback.query,
      }))
      setSaved(true)
      toast.success("Passage saved to workspace")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save passage")
    }
  }

  async function toggleFeedback(nextJudgment: "relevant" | "wrong_passage") {
    setSavingFeedback(true)
    try {
      if (judgment === nextJudgment && feedbackId !== null) {
        await api(`/quality/api/feedback/${feedbackId}`, { method: "DELETE" })
        setJudgment(null)
        setFeedbackId(null)
        toast.success("Quality feedback removed")
        return
      }

      const response = await api<{ feedback: { id: number } }>(
        "/quality/api/feedback",
        jsonRequest("POST", { ...result.feedback, judgment: nextJudgment }),
      )
      setJudgment(nextJudgment)
      setFeedbackId(response.feedback.id)
      toast.success("Quality feedback saved")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save quality feedback")
    } finally {
      setSavingFeedback(false)
    }
  }

  return (
    <Card className="animate-in overflow-hidden bg-card/85">
      <CardContent className="p-0">
        <div className="flex gap-4 p-5 md:p-6">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="m-0 truncate font-semibold">{result.document_title}</p>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                  {result.section && <span>{result.section}</span>}
                  {result.page_label && <><span aria-hidden>·</span><span>{result.page_label}</span></>}
                  {result.content_type && <><span aria-hidden>·</span><span>{result.content_type}</span></>}
                </div>
              </div>
              <Badge variant="secondary">#{result.result_rank}</Badge>
            </div>
            <p className="mt-5 whitespace-pre-wrap text-[15px] leading-7 text-foreground/90">{result.excerpt}</p>
            {showDebug && (
              <div className="mt-4 grid grid-cols-2 gap-2 rounded-lg border bg-background/45 p-3 text-xs sm:grid-cols-4">
                <div><span className="text-muted-foreground">Dense rank</span><p className="m-0 mt-1 font-medium tabular-nums">{result.dense_rank ?? "—"}</p></div>
                <div><span className="text-muted-foreground">Lexical rank</span><p className="m-0 mt-1 font-medium tabular-nums">{result.lexical_rank ?? "—"}</p></div>
                <div><span className="text-muted-foreground">Fusion rank</span><p className="m-0 mt-1 font-medium tabular-nums">{result.retrieval_rank}</p></div>
                <div><span className="text-muted-foreground">Reranker rank</span><p className="m-0 mt-1 font-medium tabular-nums">{result.rerank_rank}</p></div>
                <div><span className="text-muted-foreground">Raw logit</span><p className="m-0 mt-1 font-medium tabular-nums">{result.rerank_logit.toFixed(4)}</p></div>
                <div><span className="text-muted-foreground">Final blend</span><p className="m-0 mt-1 font-medium tabular-nums">{result.final_score.toFixed(6)}</p></div>
                <div><span className="text-muted-foreground">Tokens</span><p className="m-0 mt-1 font-medium tabular-nums">{result.rerank_token_count}</p></div>
                <div><span className="text-muted-foreground">Truncated</span><p className="m-0 mt-1 font-medium">{result.rerank_truncated ? "Yes" : "No"}</p></div>
              </div>
            )}
            <div className="mt-5 flex flex-wrap items-center gap-2">
              {result.citation_url && (
                <Button asChild size="sm"><a href={result.citation_url}>Open source</a></Button>
              )}
              <Button size="sm" variant="outline" onClick={() => copyText(result.citation_label, "Citation copied")}>Citation</Button>
              <Button size="sm" variant="outline" onClick={() => copyText(result.excerpt, "Passage copied")}>Passage</Button>
              <Button size="sm" variant="ghost" onClick={savePassage} disabled={saved}>{saved ? "Saved" : "Save"}</Button>
              <Button
                size="sm"
                variant={judgment === "relevant" ? "secondary" : "ghost"}
                className="ml-2 w-9 px-2"
                aria-label="Mark result relevant"
                aria-pressed={judgment === "relevant"}
                title="Relevant"
                disabled={savingFeedback}
                onClick={() => toggleFeedback("relevant")}
              >
                👍
              </Button>
              <Button
                size="sm"
                variant={judgment === "wrong_passage" ? "secondary" : "ghost"}
                className="w-9 px-2"
                aria-label="Mark result not relevant"
                aria-pressed={judgment === "wrong_passage"}
                title="Not relevant"
                disabled={savingFeedback}
                onClick={() => toggleFeedback("wrong_passage")}
              >
                👎
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
