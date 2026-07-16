import { cn } from "@/lib/utils"

export function Switch({ checked, onCheckedChange, disabled, label }: { checked: boolean; onCheckedChange: (checked: boolean) => void; disabled?: boolean; label?: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onCheckedChange(!checked)} className="inline-flex items-center gap-2 text-sm text-muted-foreground disabled:opacity-50">
      <span className={cn("relative inline-block h-5 w-9 rounded-full border transition-colors", checked ? "border-primary bg-primary" : "bg-muted")}>
        <span className={cn("absolute top-0.5 size-3.5 rounded-full bg-white shadow transition-transform", checked ? "translate-x-[17px]" : "translate-x-0.5")} />
      </span>
      {label}
    </button>
  )
}
