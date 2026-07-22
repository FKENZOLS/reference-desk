import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react"
import { Toaster } from "sonner"
import { AppShell } from "@/components/app-shell"
import { DocumentsPage } from "@/pages/documents-page"
import { ExperimentsPage } from "@/pages/experiments-page"
import { QualityPage } from "@/pages/quality-page"
import { SearchPage } from "@/pages/search-page"
import { UpdatesPage } from "@/pages/updates-page"
import { WorkspacePage } from "@/pages/workspace-page"

function isViewerUrl(url: URL): boolean {
  return (
    url.origin === window.location.origin &&
    url.pathname.startsWith("/viewer/")
  )
}

export default function App() {
  const [viewerUrl, setViewerUrl] = useState<string | null>(null)
  const viewerRef = useRef<HTMLIFrameElement | null>(null)

  const path =
    window.location.pathname.replace(/\/+$/, "") || "/"

  const page =
    path === "/documents"
      ? <DocumentsPage />
      : path === "/workspace"
        ? <WorkspacePage />
        : path === "/experiments"
        ? <ExperimentsPage />
        : path === "/updates"
          ? <UpdatesPage />
        : path === "/quality"
          ? <QualityPage />
          : <SearchPage />

  const closeViewer = useCallback(() => {
    setViewerUrl(null)
  }, [])

  /*
   * Intercept links to /viewer/... before the browser leaves the React page.
   *
   * SearchPage stays mounted, so its query, results, filters, expanded
   * sections, and scroll position remain in memory.
   */
  useEffect(() => {
    function handleClick(event: MouseEvent) {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey ||
        event.metaKey
      ) {
        return
      }

      const target = event.target

      if (!(target instanceof Element)) {
        return
      }

      const link = target.closest<HTMLAnchorElement>("a[href]")

      if (
        !link ||
        link.target === "_blank" ||
        link.hasAttribute("download")
      ) {
        return
      }

      let url: URL

      try {
        url = new URL(link.href, window.location.href)
      } catch {
        return
      }

      if (!isViewerUrl(url)) {
        return
      }

      event.preventDefault()

      setViewerUrl(
        `${url.pathname}${url.search}${url.hash}`,
      )
    }

    document.addEventListener("click", handleClick, true)

    return () => {
      document.removeEventListener(
        "click",
        handleClick,
        true,
      )
    }
  }, [])

  /*
   * Close the viewer with Escape.
   */
  useEffect(() => {
    if (!viewerUrl) {
      return
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault()
        closeViewer()
      }
    }

    const previousOverflow =
      document.body.style.overflow

    document.body.style.overflow = "hidden"
    window.addEventListener("keydown", handleKeyDown)

    return () => {
      document.body.style.overflow = previousOverflow

      window.removeEventListener(
        "keydown",
        handleKeyDown,
      )
    }
  }, [viewerUrl, closeViewer])

  /*
   * The Python viewer already has an element with class="back".
   *
   * Replace its href="/" or history.back() behavior with closing the iframe.
   * Since the viewer is served by the same application, its document can be
   * accessed from the parent page.
   */
  const connectViewerBackButton = useCallback(() => {
    const iframe = viewerRef.current

    if (!iframe) {
      return
    }

    try {
      const viewerDocument = iframe.contentDocument

      if (!viewerDocument) {
        return
      }

      const backButton =
        viewerDocument.querySelector<HTMLElement>(".back")

      if (!backButton) {
        return
      }

      backButton.removeAttribute("onclick")

      if (backButton instanceof HTMLAnchorElement) {
        backButton.href = "#"
      }

      backButton.setAttribute("role", "button")
      backButton.setAttribute(
        "aria-label",
        "Back to search results",
      )

      backButton.onclick = (event) => {
        event.preventDefault()
        event.stopPropagation()
        closeViewer()
      }
    } catch (error) {
      console.error(
        "Could not connect the viewer back button:",
        error,
      )
    }
  }, [closeViewer])

  return (
    <>
      <AppShell>{page}</AppShell>

      {viewerUrl && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Citation viewer"
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            background: "var(--background)",
          }}
        >
          <iframe
            ref={viewerRef}
            src={viewerUrl}
            title="Citation viewer"
            onLoad={connectViewerBackButton}
            style={{
              display: "block",
              width: "100%",
              height: "100%",
              border: 0,
              background: "var(--background)",
            }}
          />

          <button
            type="button"
            onClick={closeViewer}
            aria-label="Close citation viewer"
            title="Close citation viewer"
            style={{
              position: "fixed",
              top: "12px",
              right: "18px",
              zIndex: 1001,
              width: "36px",
              height: "36px",
              display: "grid",
              placeItems: "center",
              border: "1px solid var(--border)",
              borderRadius: "9999px",
              background: "var(--popover)",
              color: "var(--foreground)",
              boxShadow:
                "0 4px 12px rgba(0, 0, 0, 0.35)",
              cursor: "pointer",
              fontSize: "22px",
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>
      )}

      <Toaster
        theme="dark"
        richColors
        position="bottom-right"
      />
    </>
  )
}
