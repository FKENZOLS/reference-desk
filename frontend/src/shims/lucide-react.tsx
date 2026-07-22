import type { ReactElement, SVGProps } from "react"

export type LucideIcon = (props: SVGProps<SVGSVGElement>) => ReactElement
const icon = (paths: ReactElement): LucideIcon => function Icon({ className, ...props }) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true" {...props}>{paths}</svg>
}
const documentShape = <><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></>
const searchShape = <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>
const bookShape = <><path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v18H7.5A3.5 3.5 0 0 0 4 23z"/><path d="M20 5.5A3.5 3.5 0 0 0 16.5 2H13v18h3.5A3.5 3.5 0 0 1 20 23z"/></>
const arrowsShape = <><path d="M4 7h13M14 4l3 3-3 3M20 17H7M10 14l-3 3 3 3"/></>
const checkShape = <path d="m5 12 4 4L19 6"/>
const alertShape = <><circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 17h.01"/></>
const chartShape = <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></>
const panelShape = <><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/></>
const xShape = <><path d="m6 6 12 12M18 6 6 18"/></>
const trashShape = <><path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14"/></>

export const Search = icon(searchShape)
export const Filter = icon(<path d="M4 5h16l-6 7v6l-4 2v-8z"/>)
export const SlidersHorizontal = icon(<><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></>)
export const BookOpenText = icon(bookShape)
export const BookOpen = icon(bookShape)
export const Library = icon(<><path d="M4 4h4v16H4zM10 4h4v16h-4zM16 6h4v14h-4z"/></>)
export const Gauge = icon(chartShape)
export const Activity = icon(chartShape)
export const PanelLeft = icon(panelShape)
export const FileText = icon(documentShape)
export const FilePlus2 = icon(documentShape)
export const Copy = icon(<><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>)
export const ExternalLink = icon(<><path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v6H5V6h6"/></>)
export const BookmarkPlus = icon(<><path d="M6 3h12v18l-6-4-6 4zM12 7v6M9 10h6"/></>)
export const Check = icon(checkShape)
export const CheckSquare = icon(checkShape)
export const CheckCircle2 = icon(<><circle cx="12" cy="12" r="9"/>{checkShape}</>)
export const XCircle = icon(<><circle cx="12" cy="12" r="9"/>{xShape}</>)
export const X = icon(xShape)
export const CircleAlert = icon(alertShape)
export const Info = icon(alertShape)
export const Bell = icon(<><path d="M6 9a6 6 0 0 1 12 0c0 7 3 7 3 9H3c0-2 3-2 3-9"/><path d="M10 21h4"/></>)
export const Flag = icon(<><path d="M5 21V4M5 5h12l-2 4 2 4H5"/></>)
export const ThumbsUp = icon(<><path d="M7 10v10H3V10zM7 18h9l3-8h-6l1-6-3 1-4 7"/></>)
export const ThumbsDown = icon(<><path d="M7 14V4H3v10zM7 6h9l3 8h-6l1 6-3-1-4-7"/></>)
export const ChevronDown = icon(<path d="m7 9 5 5 5-5"/>)
export const LoaderCircle = icon(<path d="M21 12a9 9 0 1 1-6-8.5"/>)
export const Upload = icon(<><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></>)
export const Download = icon(<><path d="M12 4v12M7 11l5 5 5-5M4 20h16"/></>)
export const RefreshCw = icon(arrowsShape)
export const Undo2 = icon(<><path d="M9 7 4 12l5 5"/><path d="M4 12h10a6 6 0 0 1 6 6"/></>)
export const RotateCcw = icon(arrowsShape)
export const ArchiveRestore = icon(arrowsShape)
export const FolderOpen = icon(<path d="M3 6h7l2 2h9l-2 11H4z"/>)
export const FolderPlus = icon(<><path d="M3 6h7l2 2h9v11H3z"/><path d="M12 11v5M9.5 13.5h5"/></>)
export const Pencil = icon(<path d="m4 20 4-1 11-11-3-3L5 16z"/>)
export const Trash2 = icon(trashShape)
export const Clock3 = icon(<><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>)
export const Columns2 = icon(<><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></>)
export const History = icon(arrowsShape)
export const NotebookPen = icon(<><path d="M5 3h14v18H5zM8 3v18M11 15l5-5 2 2-5 5-3 1z"/></>)
export const ShieldCheck = icon(<><path d="M12 3 4 6v6c0 5 3 8 8 10 5-2 8-5 8-10V6z"/>{checkShape}</>)
export const Pause = icon(<><path d="M8 5v14M16 5v14"/></>)
export const Play = icon(<path d="m8 5 11 7-11 7z"/>)
export const HardDrive = icon(<><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 15h.01M11 15h6"/></>)
export const Database = icon(<><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></>)
export const Archive = icon(<><path d="M4 7h16v13H4zM3 4h18v3H3zM9 11h6"/></>)
export const ShieldAlert = icon(<><path d="M12 3 4 6v6c0 5 3 8 8 10 5-2 8-5 8-10V6z"/><path d="M12 8v5M12 17h.01"/></>)
export const FlaskConical = icon(<><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/><path d="M7.5 16h9"/></>)
export const Star = icon(<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9z"/>)
