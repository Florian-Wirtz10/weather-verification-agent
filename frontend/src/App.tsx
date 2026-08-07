import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { ArchiveView } from './components/ArchiveView'
import { ProgressView } from './components/ProgressView'
import { Shell } from './components/Shell'
import { TaskComposer } from './components/TaskComposer'
import type { Job, TaskPayload, ThemeName, ViewName } from './types'

const ResultView = lazy(() => import('./components/ResultView').then((module) => ({ default: module.ResultView })))

export default function App() {
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

  return (
    <Shell view={view} theme={theme} locatorBbox={locatorBbox} hasJob={Boolean(job)} hasResult={Boolean(job?.result)} onNavigate={setView} onThemeChange={setTheme}>
      {view === 'task' && <TaskComposer onSubmit={submit} onBboxChange={setLocatorBbox} />}
      {view === 'progress' && job && <ProgressView job={job} onJob={setJob} onComplete={completeJob} />}
      {view === 'result' && job && (
        <Suspense fallback={<div className="empty-state">正在加载交互式结果视图…</div>}>
          <ResultView job={job} theme={theme} />
        </Suspense>
      )}
      {view === 'archive' && <ArchiveView onOpen={openJob} />}
    </Shell>
  )
}
