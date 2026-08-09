import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import type { ThemeName } from '../types'
import { PlatformMark } from './PlatformMark'

const ChinaLocatorMap = lazy(() => import('./ChinaLocatorMap').then((module) => ({ default: module.ChinaLocatorMap })))

interface IntroTransitionProps {
  theme: ThemeName
  bbox: number[]
  onSettling: () => void
  onComplete: () => void
}

type IntroStyle = CSSProperties & {
  '--intro-mark-dx': string
  '--intro-mark-dy': string
  '--intro-mark-scale': number
  '--intro-map-dx': string
  '--intro-map-dy': string
  '--intro-map-scale': number
}

export function IntroTransition({ theme, bbox, onSettling, onComplete }: IntroTransitionProps) {
  const [ready, setReady] = useState(false)
  const markSourceRef = useRef<HTMLDivElement | null>(null)
  const mapSourceRef = useRef<HTMLDivElement | null>(null)
  const [target, setTarget] = useState({
    markDx: 0,
    markDy: 0,
    markScale: 0.28,
    mapDx: 0,
    mapDy: 0,
    mapScale: 1,
  })
  const reducedMotion = useMemo(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    [],
  )

  const measureTarget = useCallback(() => {
    const brandMark = document.querySelector<HTMLElement>('[data-brand-mark]')
    const locatorMap = document.querySelector<HTMLElement>('[data-locator-map]')
    const markSource = markSourceRef.current
    const mapSource = mapSourceRef.current
    if (!brandMark || !locatorMap || !markSource || !mapSource) return false
    const markRect = brandMark.getBoundingClientRect()
    const mapRect = locatorMap.getBoundingClientRect()
    const markSourceRect = markSource.getBoundingClientRect()
    const mapSourceRect = mapSource.getBoundingClientRect()
    if (
      markRect.width <= 0
      || mapRect.width <= 0
      || markSourceRect.width <= 0
      || mapSourceRect.width <= 0
    ) return false
    setTarget({
      markDx: markRect.left + markRect.width / 2 - (markSourceRect.left + markSourceRect.width / 2),
      markDy: markRect.top + markRect.height / 2 - (markSourceRect.top + markSourceRect.height / 2),
      markScale: markRect.width / markSourceRect.width,
      mapDx: mapRect.left + mapRect.width / 2 - (mapSourceRect.left + mapSourceRect.width / 2),
      mapDy: mapRect.top + mapRect.height / 2 - (mapSourceRect.top + mapSourceRect.height / 2),
      mapScale: mapRect.width / mapSourceRect.width,
    })
    return true
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
    let frame = 0
    const waitForLayout = () => {
      if (measureTarget()) {
        frame = window.requestAnimationFrame(() => setReady(true))
        return
      }
      frame = window.requestAnimationFrame(waitForLayout)
    }
    frame = window.requestAnimationFrame(waitForLayout)
    return () => window.cancelAnimationFrame(frame)
  }, [measureTarget])

  useEffect(() => {
    if (!ready) return
    if (reducedMotion) {
      onSettling()
      const complete = window.setTimeout(onComplete, 180)
      return () => window.clearTimeout(complete)
    }
    const settling = window.setTimeout(onSettling, 1050)
    const complete = window.setTimeout(onComplete, 2850)
    return () => {
      window.clearTimeout(settling)
      window.clearTimeout(complete)
    }
  }, [onComplete, onSettling, ready, reducedMotion])

  const style: IntroStyle = {
    '--intro-mark-dx': `${target.markDx}px`,
    '--intro-mark-dy': `${target.markDy}px`,
    '--intro-mark-scale': target.markScale,
    '--intro-map-dx': `${target.mapDx}px`,
    '--intro-map-dy': `${target.mapDy}px`,
    '--intro-map-scale': target.mapScale,
  }

  return (
    <div
      className={`intro-transition theme-${theme} ${ready ? 'ready' : ''} ${reducedMotion ? 'reduced-motion' : ''}`}
      role="dialog"
      aria-modal="true"
      style={style}
      aria-label="正在进入强对流预报智能检验平台"
    >
      <div className="intro-backdrop" aria-hidden="true" />
      <div className="intro-atmosphere" aria-hidden="true" />
      <div className="intro-brand-lockup" ref={markSourceRef}>
        <div className="intro-mark-shell"><PlatformMark title="强对流预报智能检验平台" /></div>
      </div>
      <div className="intro-wordmark">
        <strong>强对流预报智能检验平台</strong>
        <span>SEVERE CONVECTION VERIFICATION INTELLIGENCE</span>
      </div>
      <div className="intro-map-lockup" ref={mapSourceRef} aria-hidden="true">
        <Suspense fallback={<div className="locator-map locator-map-loading" />}>
          <ChinaLocatorMap bbox={bbox} theme={theme} />
        </Suspense>
      </div>
    </div>
  )
}
