import { BookmarkPlus, Check, Copy, ExternalLink, FileText, Flag, ThumbsDown, ThumbsUp } from "lucide-react"
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

export function SearchResultCard({ result }: { result: SearchResult }) {
  const [saved, setSaved] = useState(false)
  const [judgment, setJudgment] = useState<string>("")

  async function sendFeedback(next: "relevant" | "wrong_passage" | "wrong_document") {
    try {
      await api("/quality/api/feedback", jsonRequest("POST", { ...result.feedback, judgment: next }))
      setJudgment(next)
      toast.success("Feedback saved")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save feedback")
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
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <span className="mr-1">Useful?</span>
            <Button size="sm" variant={judgment === "relevant" ? "secondary" : "ghost"} onClick={() => sendFeedback("relevant")}><ThumbsUp /> Relevant</Button>
            <Button size="sm" variant={judgment === "wrong_passage" ? "secondary" : "ghost"} onClick={() => sendFeedback("wrong_passage")}><ThumbsDown /> Passage</Button>
            <Button size="sm" variant={judgment === "wrong_document" ? "secondary" : "ghost"} onClick={() => sendFeedback("wrong_document")}><Flag /> Document</Button>
          </div>
          <span className="text-[11px] tabular-nums text-muted-foreground">score {result.rerank_probability.toFixed(3)}</span>
        </div>
      </CardContent>
    </Card>
  )
}
