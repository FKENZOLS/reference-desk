export type SearchSource = { label: string; value: string }

export type SearchFeedback = {
  query: string
  chunk_id: string
  rerank_logit: number | null
  [key: string]: unknown
}

export type SearchResult = {
  chunk_id: string
  source_id: string
  document_title: string
  page_start: number | null
  page_end: number | null
  page_label: string
  section: string
  content_type: string
  document_date: string
  excerpt: string
  citation_label: string
  citation_url: string
  result_rank: number
  rerank_logit: number
  rerank_probability: number
  final_score: number
  retrieval_rank: number
  rerank_rank: number
  dense_rank: number | null
  lexical_rank: number | null
  rerank_truncated: boolean
  feedback: SearchFeedback
}

export type SearchResponse = {
  query: string
  results: SearchResult[]
  additional_results: SearchResult[]
  result_ids: string[]
  metrics: {
    dense_seconds: number
    lexical_seconds: number
    rerank_seconds: number
    total_seconds: number
    reranker_truncation_rate: number
    best_rerank_logit: number | null
    considered_count: number
  }
  gate: {
    active: boolean
    source: string
    threshold: number | null
    no_strong_evidence: boolean
  }
  query_feedback: SearchFeedback
}

export type ManagedDocument = {
  source_id: string
  name: string
  folder: string
  size: number
  modified_at: string | number
  pages: number | null
  chunks: number | null
  status: string
  file_hash?: string
  revision_count?: number
}

export type TrashDocument = ManagedDocument & { trash_id: string }

export type QuarantinedDocument = {
  quarantine_id: string
  source_id: string
  quarantined_at: string
  error: string
  file_hash: string
  size: number
}

export type DocumentRevision = {
  revision_id: string
  source_id: string
  created_at: string
  file_hash: string
  replaced_by_hash: string
  size: number
}

export type QueueItem = {
  id: string
  source_id: string
  action: "ingest" | "delete"
  status: "queued" | "processing" | "complete" | "failed" | "quarantined"
  attempts: number
  enqueued_at: string
  started_at: string | null
  finished_at: string | null
  error: string
}

export type CorpusQueue = {
  paused: boolean
  pause_requested_at: string | null
  run_id: string | null
  items: QueueItem[]
  counts: Record<string, number>
  remaining: number
  updated_at: string | null
}

export type CorpusStorage = {
  documents: number
  index: number
  workspace: number
  trash: number
  quarantine: number
  revisions: number
  backups: number
  debug: number
  active: number
  total: number
}

export type CorpusHealth = {
  status: "healthy" | "attention" | "critical"
  generated_at: string
  documents: number
  pages: number
  chunks: number
  indexed: number
  pending: number
  quarantined: number
  revisions: number
  invalid_sources: string[]
  stale_sources: string[]
  duplicate_groups: { file_hash: string; sources: string[] }[]
  issues: { severity: string; label: string; count: number; sources?: string[] }[]
  storage: CorpusStorage
}

export type CorpusBackup = {
  backup_id: string
  version: number
  label: string
  created_at: string
  filename: string
  size: number
}

export type DocumentJob = {
  running?: boolean
  status?: string
  title?: string
  message?: string
  progress?: number
  state?: string
  log?: string[]
  completed?: boolean
  force?: boolean
  pause_requested?: boolean
  [key: string]: unknown
}

export type DocumentState = {
  documents: ManagedDocument[]
  trash: TrashDocument[]
  quarantine?: QuarantinedDocument[]
  revisions?: DocumentRevision[]
  counts: { documents: number; indexed: number; pending: number; trash: number; quarantine?: number; revisions?: number }
  pending_sources?: string[]
  deleted_sources?: string[]
  queue?: CorpusQueue
  health?: CorpusHealth
  backups?: CorpusBackup[]
  job: DocumentJob
  app_instance_id: string
  hardware?: {
    requested: string
    backend: "cuda" | "rocm" | "cpu"
    torch_device: string
    available: boolean
    device_name: string
    torch_version: string
    runtime_version: string
    total_memory_bytes: number | null
  }
}

export type Collection = {
  id: number
  name: string
  description?: string
  bookmark_count?: number
}

export type Bookmark = {
  id: number
  chunk_id: string
  source_id: string
  document_title: string
  page_start: number | null
  page_end: number | null
  section: string
  content_type: string
  document_date: string
  excerpt: string
  citation_label: string
  citation_url: string
  query: string
  collection_id: number | null
  collection_name?: string
  note: string
  created_at: string
  updated_at: string
}

export type SearchHistoryItem = {
  id: number
  query: string
  source_filter: string
  section_filter: string
  content_filter: string
  date_filter: string
  within_results: boolean
  result_count: number
  created_at: string
}

export type WorkspaceState = {
  bookmarks: Bookmark[]
  collections: Collection[]
  history: SearchHistoryItem[]
}

export type Calibration = {
  threshold: number | null
  positive_count: number
  negative_count: number
  positive_recall: number | null
  specificity: number | null
  balanced_accuracy: number | null
  ready: boolean
  enabled: boolean
  active: boolean
  minimum_positive: number
  minimum_negative: number
  updated_at: string
}

export type Feedback = {
  id: number
  query: string
  judgment: "relevant" | "wrong_passage" | "wrong_document" | "no_relevant_result"
  document_title?: string
  source_id?: string
  page_start?: number | null
  section?: string
  excerpt?: string
  rerank_logit?: number | null
  created_at: string
  updated_at?: string
}

export type QualityState = {
  summary: {
    total: number
    counts: Record<string, number>
    benchmark_cases: number
    answerable_cases: number
    unanswerable_cases: number
    calibration: Calibration
  }
  feedback: Feedback[]
}
