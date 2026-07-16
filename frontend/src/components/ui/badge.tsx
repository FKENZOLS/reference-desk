import type { HTMLAttributes } from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold", {
  variants: {
    variant: {
      default: "border-transparent bg-primary/15 text-primary",
      secondary: "border-transparent bg-secondary text-secondary-foreground",
      outline: "border-border text-muted-foreground",
      success: "border-emerald-500/20 bg-emerald-500/10 text-emerald-300",
      warning: "border-amber-500/20 bg-amber-500/10 text-amber-300",
      destructive: "border-red-500/20 bg-red-500/10 text-red-300",
    },
  },
  defaultVariants: { variant: "default" },
})

export function Badge({ className, variant, ...props }: HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeVariants>) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}
