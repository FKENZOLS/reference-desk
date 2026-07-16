import { Children, cloneElement, isValidElement, type HTMLAttributes, type ReactElement } from "react"

export function Slot({ children, className, ...props }: HTMLAttributes<HTMLElement> & { children?: ReactElement }) {
  const child = Children.only(children)
  if (!isValidElement(child)) return null
  const childProps = child.props as { className?: string }
  return cloneElement(child, { ...props, ...child.props, className: [className, childProps.className].filter(Boolean).join(" ") })
}
