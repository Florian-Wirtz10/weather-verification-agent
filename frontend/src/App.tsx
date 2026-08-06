import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { ArchiveView } from './components/ArchiveView'
import { ProgressView } from './components/ProgressView'
import { Shell } from './components/Shell'
import { TaskComposer } from './components/TaskComposer'
import type { Job, TaskPayload, ViewName } from './types'

const ResultView = lazy(() => import('./components/ResultView').then((module) => ({ default: module.ResultView })))

export default function App() {
  const [view, setView] = useState<ViewName>('task')
  const [systemReady, setSystemReady] = useState(false)
  const [job, setJob] = useState<Job | null>(null)

  useEffect(() => { api.health().then(() => setSystemReady(true)).catch(() => setSystemReady(false)) }, [])

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
    <Shell view={view} systemReady={systemReady} hasJob={Boolean(job)} hasResult={Boolean(job?.result)} onNavigate={setView}>
      {view === 'task' && <TaskComposer onSubmit={submit} />}
      {view === 'progress' && job && <ProgressView job={job} onJob={setJob} onComplete={completeJob} />}
      {view === 'result' && job && (
        <Suspense fallback={<div className="empty-state">正在加载交互式结果视图…</div>}>
          <ResultView job={job} />
        </Suspense>
      )}
      {view === 'archive' && <ArchiveView onOpen={openJob} />}
    </Shell>
  )
}
