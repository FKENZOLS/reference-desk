import { BookOpen, CheckSquare, Clock3, Columns2, Download, ExternalLink, FileText, FolderPlus, History, NotebookPen, Search, Trash2 } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { EmptyState } from "@/components/empty-state"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Modal } from "@/components/ui/modal"
import { Textarea } from "@/components/ui/textarea"
import { api, jsonRequest } from "@/lib/api"
import { formatDate } from "@/lib/utils"
import type { Bookmark, Collection, WorkspaceState } from "@/types"

const emptyState: WorkspaceState = { bookmarks: [], collections: [], history: [] }

function pageLabel(bookmark: Bookmark) {
  if (!bookmark.page_start) return ""
  return bookmark.page_end && bookmark.page_end !== bookmark.page_start ? `pages ${bookmark.page_start}–${bookmark.page_end}` : `page ${bookmark.page_start}`
}

function historyUrl(item: WorkspaceState["history"][number]) {
  const params = new URLSearchParams({ q: item.query })
  if (item.source_filter) params.set("source", item.source_filter)
  if (item.section_filter) params.set("section", item.section_filter)
  if (item.content_filter) params.set("type", item.content_filter)
  if (item.date_filter) params.set("date", item.date_filter)
  return `/?${params}`
}

export function WorkspacePage() {
  const [state, setState] = useState<WorkspaceState>(emptyState)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState("")
  const [collectionFilter, setCollectionFilter] = useState("")
  const [selected, setSelected] = useState<number[]>([])
  const [editing, setEditing] = useState<Bookmark | null>(null)
  const [note, setNote] = useState("")
  const [editCollection, setEditCollection] = useState("")
  const [collectionModal, setCollectionModal] = useState(false)
  const [collectionName, setCollectionName] = useState("")
  const [collectionDescription, setCollectionDescription] = useState("")
  const [compareOpen, setCompareOpen] = useState(false)

  async function refresh() {
    try { setState(await api<WorkspaceState>("/workspace/api/state")) }
    catch (error) { toast.error(error instanceof Error ? error.message : "Could not load workspace") }
    finally { setLoading(false) }
  }
  useEffect(() => { void refresh() }, [])

  const bookmarks = useMemo(() => {
    const needle = query.toLowerCase().trim()
    return state.bookmarks.filter((bookmark) => {
      const inCollection = !collectionFilter || String(bookmark.collection_id || "") === collectionFilter
      const matches = !needle || `${bookmark.document_title} ${bookmark.section} ${bookmark.excerpt} ${bookmark.note}`.toLowerCase().includes(needle)
      return inCollection && matches
    })
  }, [collectionFilter, query, state.bookmarks])
  const selectedBookmarks = state.bookmarks.filter((bookmark) => selected.includes(bookmark.id))

  function toggleSelected(id: number) {
    setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
  }

  function openEdit(bookmark: Bookmark) {
    setEditing(bookmark)
    setNote(bookmark.note || "")
    setEditCollection(bookmark.collection_id ? String(bookmark.collection_id) : "")
  }

  async function saveEdit() {
    if (!editing) return
    try {
      await api(`/workspace/api/bookmarks/${editing.id}`, jsonRequest("PATCH", { note, collection_id: editCollection ? Number(editCollection) : null }))
      setEditing(null)
      toast.success("Passage updated")
      await refresh()
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not update passage") }
  }

  async function deleteBookmark(id: number) {
    if (!window.confirm("Remove this saved passage?")) return
    try {
      await api(`/workspace/api/bookmarks/${id}`, { method: "DELETE" })
      setSelected((current) => current.filter((item) => item !== id))
      toast.success("Passage removed")
      await refresh()
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not remove passage") }
  }

  async function createCollection() {
    try {
      const created = await api<Collection>("/workspace/api/collections", jsonRequest("POST", { name: collectionName, description: collectionDescription }))
      setCollectionModal(false)
      setCollectionName("")
      setCollectionDescription("")
      toast.success(`Collection “${created.name}” created`)
      await refresh()
    } catch (error) { toast.error(error instanceof Error ? error.message : "Could not create collection") }
  }

  function exportSelection(format: "markdown" | "word") {
    if (!selected.length) return toast.error("Select at least one passage")
    window.location.href = `/workspace/export/${format}?ids=${selected.join(",")}`
  }

  return (
    <div>
      <PageHeader title="Research workspace" actions={<>
        <Button variant="outline" onClick={() => setCollectionModal(true)}><FolderPlus /> New collection</Button>
        <Button variant="outline" onClick={() => exportSelection("markdown")} disabled={!selected.length}><Download /> Markdown</Button>
        <Button onClick={() => exportSelection("word")} disabled={!selected.length}><Download /> Word</Button>
      </>} />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
        <section>
          <Card className="mb-4 bg-card/75"><CardContent className="flex flex-wrap gap-3 p-4">
            <div className="relative min-w-56 flex-1"><Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search saved passages" className="pl-9" /></div>
            <div className="select-wrap w-full sm:w-56"><select className="native-select" value={collectionFilter} onChange={(event) => setCollectionFilter(event.target.value)}><option value="">All collections</option>{state.collections.map((collection) => <option key={collection.id} value={collection.id}>{collection.name}</option>)}</select></div>
            <Button variant="outline" disabled={selected.length !== 2} onClick={() => setCompareOpen(true)}><Columns2 /> Compare{selected.length ? ` (${selected.length})` : ""}</Button>
          </CardContent></Card>

          {loading ? <div className="p-10 text-center text-muted-foreground">Loading workspace…</div> : bookmarks.length ? (
            <div className="space-y-3">
              {bookmarks.map((bookmark) => <Card key={bookmark.id} className={`bg-card/75 transition-colors ${selected.includes(bookmark.id) ? "border-primary/60" : ""}`}><CardContent className="p-5">
                <div className="flex gap-3">
                  <button onClick={() => toggleSelected(bookmark.id)} className={`mt-0.5 grid size-5 shrink-0 place-items-center rounded border ${selected.includes(bookmark.id) ? "border-primary bg-primary text-primary-foreground" : "bg-background"}`} aria-label={selected.includes(bookmark.id) ? "Deselect passage" : "Select passage"}>{selected.includes(bookmark.id) && <CheckSquare className="size-3.5" />}</button>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start justify-between gap-2"><div><p className="m-0 font-semibold">{bookmark.document_title}</p><p className="mt-1 text-xs text-muted-foreground">{[bookmark.section, pageLabel(bookmark)].filter(Boolean).join(" · ")}</p></div>{bookmark.collection_name && <Badge variant="secondary">{bookmark.collection_name}</Badge>}</div>
                    <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-foreground/88">{bookmark.excerpt}</p>
                    {bookmark.note && <div className="mt-4 rounded-lg border-l-2 border-primary bg-primary/5 px-3 py-2 text-sm"><span className="text-xs font-medium text-primary">Note</span><p className="mb-0 mt-1 whitespace-pre-wrap">{bookmark.note}</p></div>}
                    <div className="mt-4 flex flex-wrap gap-2">{bookmark.citation_url && <Button size="sm" asChild><a href={bookmark.citation_url}><ExternalLink /> Open source</a></Button>}<Button size="sm" variant="outline" onClick={() => openEdit(bookmark)}><NotebookPen /> Note & collection</Button><Button size="sm" variant="ghost" className="text-destructive" onClick={() => void deleteBookmark(bookmark.id)}><Trash2 /> Remove</Button></div>
                  </div>
                </div>
              </CardContent></Card>)}
            </div>
          ) : <EmptyState icon={BookOpen} title="No saved passages here" detail={query || collectionFilter ? "Try a different filter." : "Save useful results from Search to build your reference workspace."} />}
        </section>

        <aside className="space-y-5">
          <Card className="bg-card/70"><CardContent className="p-5"><div className="mb-4 flex items-center gap-2 font-semibold"><FolderPlus className="size-4 text-primary" /> Collections</div>{state.collections.length ? <div className="space-y-2">{state.collections.map((collection) => <button key={collection.id} className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left hover:bg-muted" onClick={() => setCollectionFilter(String(collection.id))}><span className="truncate">{collection.name}</span><Badge variant="secondary">{collection.bookmark_count || 0}</Badge></button>)}</div> : <p className="m-0 text-sm text-muted-foreground">No collections yet.</p>}</CardContent></Card>
          <Card className="bg-card/70"><CardContent className="p-5"><div className="mb-4 flex items-center gap-2 font-semibold"><History className="size-4 text-primary" /> Recent searches</div>{state.history.length ? <div className="scrollbar-thin max-h-[460px] space-y-1 overflow-auto pr-1">{state.history.slice(0, 25).map((item) => <a key={item.id} href={historyUrl(item)} className="block rounded-lg px-3 py-2.5 text-sm hover:bg-muted"><p className="m-0 line-clamp-2 font-medium">{item.query}</p><p className="mb-0 mt-1 flex items-center gap-1 text-[11px] text-muted-foreground"><Clock3 className="size-3" /> {formatDate(item.created_at)} · {item.result_count} results</p></a>)}</div> : <p className="m-0 text-sm text-muted-foreground">Search history will appear here.</p>}</CardContent></Card>
        </aside>
      </div>

      <Modal open={Boolean(editing)} title="Passage details" onClose={() => setEditing(null)}>
        <Label htmlFor="bookmark-collection">Collection</Label><div className="select-wrap mb-4"><select id="bookmark-collection" className="native-select" value={editCollection} onChange={(event) => setEditCollection(event.target.value)}><option value="">No collection</option>{state.collections.map((collection) => <option key={collection.id} value={collection.id}>{collection.name}</option>)}</select></div>
        <Label htmlFor="bookmark-note">Note</Label><Textarea id="bookmark-note" value={note} onChange={(event) => setNote(event.target.value)} rows={7} placeholder="Add your interpretation, decision, or follow-up…" />
        <div className="mt-5 flex justify-end gap-2"><Button variant="ghost" onClick={() => setEditing(null)}>Cancel</Button><Button onClick={saveEdit}>Save</Button></div>
      </Modal>

      <Modal open={collectionModal} title="New collection" onClose={() => setCollectionModal(false)}>
        <Label htmlFor="collection-name">Name</Label><Input id="collection-name" value={collectionName} onChange={(event) => setCollectionName(event.target.value)} autoFocus />
        <Label htmlFor="collection-description" className="mt-4">Description</Label><Textarea id="collection-description" value={collectionDescription} onChange={(event) => setCollectionDescription(event.target.value)} rows={4} />
        <div className="mt-5 flex justify-end gap-2"><Button variant="ghost" onClick={() => setCollectionModal(false)}>Cancel</Button><Button onClick={createCollection} disabled={!collectionName.trim()}>Create</Button></div>
      </Modal>

      <Modal open={compareOpen} title="Compare passages" onClose={() => setCompareOpen(false)} wide>
        <div className="grid gap-4 md:grid-cols-2">{selectedBookmarks.slice(0, 2).map((bookmark) => <div key={bookmark.id} className="rounded-xl border bg-background/45 p-5"><p className="m-0 font-semibold">{bookmark.document_title}</p><p className="mt-1 text-xs text-muted-foreground">{[bookmark.section, pageLabel(bookmark)].filter(Boolean).join(" · ")}</p><p className="mt-5 whitespace-pre-wrap text-sm leading-7">{bookmark.excerpt}</p>{bookmark.note && <p className="mt-4 border-l-2 border-primary pl-3 text-sm text-muted-foreground">{bookmark.note}</p>}</div>)}</div>
      </Modal>
    </div>
  )
}
