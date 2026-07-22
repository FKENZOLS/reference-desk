import { Children, cloneElement, isValidElement, type HTMLAttributes, type ReactNode } from "react"

export function Slot({ children, className, ...props }: HTMLAttributes<HTMLElement> & { children?: ReactNode }) {
  const child = Children.only(children)
  if (!isValidElement<{ className?: string }>(child)) return null
  const childProps = child.props
  return cloneElement(child, { ...props, ...child.props, className: [className, childProps.className].filter(Boolean).join(" ") })
}
