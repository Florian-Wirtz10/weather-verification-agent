import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { ArchiveView } from './components/ArchiveView'
import { IntroTransition } from './components/IntroTransition'
import { ProgressView } from './components/ProgressView'
import { Shell } from './components/Shell'
import { TaskComposer } from './components/TaskComposer'
import type { Job, TaskPayload, ThemeName, ViewName } from './types'

const ResultView = lazy(() => import('./components/ResultView').then((module) => ({ default: module.ResultView })))

export default function App() {
  const [introStage, setIntroStage] = useState<'active' | 'settling' | 'complete'>('active')
  const [view, setView] = useState<ViewName>('task')
  const [job, setJob] = useState<Job | null>(null)
  const [locatorBbox, setLocatorBbox] = useState<number[]>([29, 33.5, 108, 116.5])
  const [theme, setTheme] = useState<ThemeName>(() => {
    const saved = window.localStorage.getItem('verification-theme')
    return saved === 'light' || saved === 'dark' ? saved : 'dark'
  })

  useEffect(() => {
    window.localStorage.setItem('verification-theme', theme)
    document.documentElement.style.colorScheme = theme
  }, [theme])

  async function submit(payload: TaskPayload) {
    const created = await api.createJob(payload)
    setJob(created)
    setView('progress')
  }

  function openJob(value: Job) {
    setJob(value)
    setView(value.status === 'complete' ? 'result' : 'progress')
  }

  const completeJob = useCallback((value: Job) => {
    setJob(value)
    setView('result')
  }, [])

  const settleIntro = useCallback(() => setIntroStage((current) => (
    current === 'complete' ? current : 'settling'
  )), [])
  const completeIntro = useCallback(() => setIntroStage('complete'), [])

  return (
    <>
      <Shell view={view} theme={theme} introStage={introStage} locatorBbox={locatorBbox} hasJob={Boolean(job)} hasResult={Boolean(job?.result)} onNavigate={setView} onThemeChange={setTheme}>
        {view === 'task' && <TaskComposer onSubmit={submit} onBboxChange={setLocatorBbox} />}
        {view === 'progress' && job && <ProgressView job={job} onJob={setJob} onComplete={completeJob} />}
        {view === 'result' && job && (
          <Suspense fallback={<div className="empty-state">正在加载交互式结果视图…</div>}>
            <ResultView job={job} theme={theme} />
          </Suspense>
        )}
        {view === 'archive' && <ArchiveView onOpen={openJob} />}
      </Shell>
      {introStage !== 'complete' && (
        <IntroTransition theme={theme} bbox={locatorBbox} onSettling={settleIntro} onComplete={completeIntro} />
      )}
    </>
  )
}
