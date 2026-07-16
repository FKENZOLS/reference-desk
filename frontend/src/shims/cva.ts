type Config = { variants?: Record<string, Record<string, string>>; defaultVariants?: Record<string, string> }
export type VariantProps<T> = T extends (props?: infer P) => string ? P : never

export function cva(base: string, config: Config = {}) {
  return (props: Record<string, unknown> = {}) => {
    const classes = [base]
    for (const [name, choices] of Object.entries(config.variants || {})) {
      const value = String(props[name] ?? config.defaultVariants?.[name] ?? "")
      if (choices[value]) classes.push(choices[value])
    }
    if (props.className) classes.push(String(props.className))
    return classes.join(" ")
  }
}
