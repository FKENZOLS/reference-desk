import { ChevronDown } from "lucide-react"
import { type KeyboardEvent, useEffect, useId, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import type { SearchSource } from "@/types"

type SearchScopeSelectProps = {
  id?: string
  value: string
  options: SearchSource[]
  onChange: (value: string) => void
  placeholder?: string
}

function filterOptions(options: SearchSource[], query: string) {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return options
  return options.filter(
    (item) =>
      item.label.toLowerCase().includes(normalized) ||
      item.value.toLowerCase().includes(normalized),
  )
}

export function SearchScopeSelect({
  id,
  value,
  options,
  onChange,
  placeholder = "All documents or type to filter…",
}: SearchScopeSelectProps) {
  const listboxId = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [activeIndex, setActiveIndex] = useState(0)

  const selected = useMemo(
    () => options.find((item) => item.value === value) ?? options[0],
    [options, value],
  )

  const filtered = useMemo(() => filterOptions(options, open ? query : ""), [open, options, query])

  useEffect(() => {
    if (!open) return
    function handlePointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handlePointerDown)
    return () => document.removeEventListener("mousedown", handlePointerDown)
  }, [open])

  useEffect(() => {
    if (!open) return
    setActiveIndex(0)
  }, [open, query, options])

  useEffect(() => {
    if (value && options.length > 1 && !options.some((item) => item.value === value)) {
      onChange("")
    }
  }, [onChange, options, value])

  function openList() {
    setOpen(true)
    setQuery("")
    setActiveIndex(0)
  }

  function selectOption(option: SearchSource) {
    onChange(option.value)
    setQuery("")
    setOpen(false)
    inputRef.current?.blur()
  }

  function handleFocus() {
    openList()
  }

  function handleBlur() {
    setOpen(false)
    setQuery("")
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault()
      if (!open) openList()
      setActiveIndex((index) => Math.min(index + 1, Math.max(filtered.length - 1, 0)))
      return
    }
    if (event.key === "ArrowUp") {
      event.preventDefault()
      if (!open) openList()
      setActiveIndex((index) => Math.max(index - 1, 0))
      return
    }
    if (event.key === "Enter") {
      if (open && filtered[activeIndex]) {
        event.preventDefault()
        selectOption(filtered[activeIndex])
      }
      return
    }
    if (event.key === "Escape") {
      event.preventDefault()
      setOpen(false)
      setQuery("")
      inputRef.current?.blur()
    }
  }

  const inputValue = open ? query : (selected?.label ?? "")

  return (
    <div ref={rootRef} className="relative">
      <input
        ref={inputRef}
        id={id}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-autocomplete="list"
        autoComplete="off"
        value={inputValue}
        placeholder={placeholder}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onChange={(event) => {
          setQuery(event.target.value)
          setOpen(true)
        }}
        onKeyDown={handleKeyDown}
        className={cn(
          "flex h-10 w-full rounded-lg border border-input bg-background px-3 py-2 pr-9 text-sm outline-none",
          "placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring",
        )}
      />
      <ChevronDown className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      {open && (
        <ul
          id={listboxId}
          role="listbox"
          className="absolute z-50 mt-1 max-h-56 w-full overflow-auto rounded-lg border bg-popover p-1 shadow-2xl"
        >
          {filtered.length ? (
            filtered.map((item, index) => (
              <li key={item.value || "__all__"} role="option" aria-selected={item.value === value}>
                <button
                  type="button"
                  className={cn(
                    "w-full rounded-md px-2.5 py-2 text-left text-sm transition-colors hover:bg-accent",
                    index === activeIndex && "bg-accent text-accent-foreground",
                    item.value === value && index !== activeIndex && "text-primary",
                  )}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => selectOption(item)}
                >
                  {item.label}
                </button>
              </li>
            ))
          ) : (
            <li className="px-2.5 py-2 text-sm text-muted-foreground">No matching documents</li>
          )}
        </ul>
      )}
    </div>
  )
}
