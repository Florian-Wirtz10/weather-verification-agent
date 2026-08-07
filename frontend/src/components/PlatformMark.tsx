import { useId } from 'react'

interface PlatformMarkProps {
  className?: string
  title?: string
}

export function PlatformMark({ className = '', title }: PlatformMarkProps) {
  const titleId = useId()

  return (
    <svg
      className={`platform-mark ${className}`.trim()}
      viewBox="0 0 64 64"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      aria-labelledby={title ? titleId : undefined}
    >
      {title && <title id={titleId}>{title}</title>}
      <circle className="mark-ring mark-ring-outer" cx="32" cy="32" r="25" />
      <circle className="mark-ring mark-ring-inner" cx="32" cy="32" r="15.5" />
      <path className="mark-sweep" d="M32 32 48.8 13.5A25 25 0 0 1 56.3 38Z" />
      <path className="mark-beam" d="M32 32 49.5 12.7" />
      <path className="mark-echo mark-echo-observed" d="M17.5 39.2c4.1-9.6 9.7-14.1 16.6-13.5 5.5.5 9.8 4 12.8 10.5-5.1-2.5-9.7-2.6-13.8-.3-4.7 2.6-9.9 3.7-15.6 3.3Z" />
      <path className="mark-echo mark-echo-forecast" d="M20.1 43.8c4.8-6.8 9.3-10 13.5-9.6 4 .3 7.5 2.6 10.6 6.8-4.5-1.2-8.3-.9-11.5.9-3.7 2-7.9 2.7-12.6 1.9Z" />
      <circle className="mark-core" cx="33.1" cy="34.2" r="3.5" />
      <path className="mark-crosshair" d="M32 4v5M32 55v5M4 32h5M55 32h5" />
    </svg>
  )
}
