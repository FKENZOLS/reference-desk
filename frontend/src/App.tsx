import { Toaster } from "sonner"
import { AppShell } from "@/components/app-shell"
import { DocumentsPage } from "@/pages/documents-page"
import { QualityPage } from "@/pages/quality-page"
import { SearchPage } from "@/pages/search-page"
import { WorkspacePage } from "@/pages/workspace-page"

export default function App() {
  const path = window.location.pathname.replace(/\/+$/, "") || "/"
  const page = path === "/documents"
    ? <DocumentsPage />
    : path === "/workspace"
      ? <WorkspacePage />
      : path === "/quality"
        ? <QualityPage />
        : <SearchPage />
  return (
    <>
      <AppShell>{page}</AppShell>
      <Toaster theme="dark" richColors position="bottom-right" />
    </>
  )
}
