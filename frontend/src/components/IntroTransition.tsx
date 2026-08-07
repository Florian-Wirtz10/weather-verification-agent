import { useCallback, useEffect, useLayoutEffect, useMemo, useState, type CSSProperties } from 'react'
import type { ThemeName } from '../types'
import { PlatformMark } from './PlatformMark'

interface IntroTransitionProps {
  theme: ThemeName
  onSettling: () => void
  onComplete: () => void
}

type IntroStyle = CSSProperties & {
  '--intro-dx': string
  '--intro-dy': string
  '--intro-scale': number
}

const MARK_SIZE = 116

export function IntroTransition({ theme, onSettling, onComplete }: IntroTransitionProps) {
  const [ready, setReady] = useState(false)
  const [target, setTarget] = useState({ dx: 0, dy: 0, scale: 0.28 })
  const reducedMotion = useMemo(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    [],
  )

  const measureTarget = useCallback(() => {
    const brandMark = document.querySelector<HTMLElement>('[data-brand-mark]')
    if (!brandMark) return
    const rect = brandMark.getBoundingClientRect()
    setTarget({
      dx: rect.left + rect.width / 2 - window.innerWidth / 2,
      dy: rect.top + rect.height / 2 - window.innerHeight / 2,
      scale: rect.width / MARK_SIZE,
    })
  }, [])

  useLayoutEffect(() => {
    document.body.classList.add('intro-locked')
    measureTarget()
    window.addEventListener('resize', measureTarget)
    return () => {
      document.body.classList.remove('intro-locked')
      window.removeEventListener('resize', measureTarget)
    }
  }, [measureTarget])

  useEffect(() => {
    const fallback = window.setTimeout(() => setReady(true), 650)
    return () => window.clearTimeout(fallback)
  }, [])

  useEffect(() => {
    if (!ready) return
    if (reducedMotion) {
      onSettling()
      const complete = window.setTimeout(onComplete, 180)
      return () => window.clearTimeout(complete)
    }
    const settling = window.setTimeout(onSettling, 720)
    const complete = window.setTimeout(onComplete, 1750)
    return () => {
      window.clearTimeout(settling)
      window.clearTimeout(complete)
    }
  }, [onComplete, onSettling, ready, reducedMotion])

  const style: IntroStyle = {
    '--intro-dx': `${target.dx}px`,
    '--intro-dy': `${target.dy}px`,
    '--intro-scale': target.scale,
  }

  return (
    <div
      className={`intro-transition theme-${theme} ${ready ? 'ready' : ''} ${reducedMotion ? 'reduced-motion' : ''}`}
      role="dialog"
      aria-modal="true"
      style={style}
      aria-label="正在进入强对流预报智能检验平台"
    >
      <img
        className="intro-backdrop"
        src="/weather-backdrop.jpg"
        alt=""
        onLoad={() => setReady(true)}
        aria-hidden="true"
      />
      <div className="intro-atmosphere" aria-hidden="true" />
      <div className="intro-brand-lockup">
        <div className="intro-mark-shell"><PlatformMark title="强对流预报智能检验平台" /></div>
      </div>
      <div className="intro-wordmark">
        <strong>强对流预报智能检验平台</strong>
        <span>SEVERE CONVECTION VERIFICATION INTELLIGENCE</span>
      </div>
      <button className="intro-skip" type="button" onClick={onComplete}>跳过动画</button>
    </div>
  )
}
