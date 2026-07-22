import { BookmarkPlus, Check, Copy, ExternalLink, FileText, Flag, Info, ThumbsDown, ThumbsUp, Undo2 } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api, jsonRequest } from "@/lib/api"
import type { SearchResult } from "@/types"

const copyText = async (value: string, message: string) => {
  await navigator.clipboard.writeText(value)
  toast.success(message)
}

export function SearchResultCard({ result, showDebug = false, selected = false, onSelected }: { result: SearchResult; showDebug?: boolean; selected?: boolean; onSelected?: (selected: boolean) => void }) {
  const [saved, setSaved] = useState(false)
  const [judgment, setJudgment] = useState<string>("")
  const [reason, setReason] = useState("")
  const [feedbackId, setFeedbackId] = useState<number | null>(null)

  async function sendFeedback(next: "relevant" | "wrong_passage" | "wrong_document") {
    try {
      const response = await api<{ feedback: { id: number } }>("/quality/api/feedback", jsonRequest("POST", { ...result.feedback, judgment: next, reason }))
      setJudgment(next)
      setFeedbackId(response.feedback.id)
      toast.success("Feedback saved")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save feedback")
    }
  }

  async function undoFeedback() {
    if (!feedbackId) return
    try {
      await api(`/quality/api/feedback/${feedbackId}`, { method: "DELETE" })
      setJudgment("")
      setFeedbackId(null)
      toast.success("Feedback removed")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not undo feedback")
    }
  }

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

  return (
    <Card className="animate-in overflow-hidden bg-card/85">
      <CardContent className="p-0">
        <div className="flex gap-4 p-5 md:p-6">
          <div className="hidden size-9 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary sm:grid">
            <FileText className="size-4" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">{onSelected && <input type="checkbox" checked={selected} onChange={(event) => onSelected(event.target.checked)} aria-label={`Select ${result.document_title}`} className="size-4 accent-primary" />}<p className="m-0 truncate font-semibold">{result.document_title}</p></div>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                  {result.section && <span>{result.section}</span>}
                  {result.page_label && <><span aria-hidden>·</span><span>{result.page_label}</span></>}
                  {result.content_type && <><span aria-hidden>·</span><span>{result.content_type}</span></>}
                </div>
              </div>
              <Badge variant="secondary">#{result.result_rank}</Badge>
            </div>
            <p className="mt-5 whitespace-pre-wrap text-[15px] leading-7 text-foreground/90">{result.excerpt}</p>
            <details className="mt-4 rounded-lg border bg-background/35 px-3 py-2 text-xs text-muted-foreground">
              <summary className="flex cursor-pointer list-none items-center gap-2 font-medium text-foreground/80"><Info className="size-3.5" /> Why this result</summary>
              <p className="mb-0 mt-2">Found by {result.explanation.found_by}. Fusion position {result.explanation.fusion_position ?? "—"}; reranker position {result.explanation.reranker_position ?? "—"}. {result.explanation.selected_because}.</p>
            </details>
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
                <Button asChild size="sm"><a href={result.citation_url}><ExternalLink /> Open source</a></Button>
              )}
              <Button size="sm" variant="outline" onClick={() => copyText(result.citation_label, "Citation copied")}><Copy /> Citation</Button>
              <Button size="sm" variant="outline" onClick={() => copyText(result.excerpt, "Passage copied")}><Copy /> Passage</Button>
              <Button size="sm" variant="ghost" onClick={savePassage} disabled={saved}>{saved ? <Check /> : <BookmarkPlus />}{saved ? "Saved" : "Save"}</Button>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border-t bg-muted/25 px-5 py-3 md:px-6">
          <div className="flex flex-1 flex-wrap items-center gap-1 text-xs text-muted-foreground">
            <span className="mr-1">Useful?</span>
            <Button size="sm" variant={judgment === "relevant" ? "secondary" : "ghost"} onClick={() => sendFeedback("relevant")}><ThumbsUp /> Relevant</Button>
            <Button size="sm" variant={judgment === "wrong_passage" ? "secondary" : "ghost"} onClick={() => sendFeedback("wrong_passage")}><ThumbsDown /> Passage</Button>
            <Button size="sm" variant={judgment === "wrong_document" ? "secondary" : "ghost"} onClick={() => sendFeedback("wrong_document")}><Flag /> Document</Button>
            {feedbackId && <Button size="sm" variant="ghost" onClick={undoFeedback}><Undo2 /> Undo</Button>}
            <Input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Optional: explain why" className="ml-1 h-8 min-w-48 flex-1 text-xs" />
          </div>
          <span className="text-[11px] tabular-nums text-muted-foreground">score {result.rerank_probability.toFixed(3)}</span>
        </div>
      </CardContent>
    </Card>
  )
}
