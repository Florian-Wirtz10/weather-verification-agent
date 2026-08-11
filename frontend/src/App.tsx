import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { ArchiveView } from './components/ArchiveView'
import { DataCatalogView } from './components/DataCatalogView'
import { IntroTransition } from './components/IntroTransition'
import { ModelPromptView } from './components/ModelPromptView'
import { ProgressView } from './components/ProgressView'
import { Shell } from './components/Shell'
import { TaskComposer } from './components/TaskComposer'
import type { Job, TaskPayload, ThemeName, ViewName } from './types'

const ResultView = lazy(() => import('./components/ResultView').then((module) => ({ default: module.ResultView })))

export default function App() {
  const [introStage, setIntroStage] = useState<'active' | 'settling' | 'complete'>('active')
  const [view, setView] = useState<ViewName>('task')
  const [job, setJob] = useState<Job | null>(null)
  const [taskDraft, setTaskDraft] = useState<Partial<TaskPayload> | undefined>()
  const [taskSession, setTaskSession] = useState(0)
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

  function newTask(draft?: Partial<TaskPayload>) {
    setJob(null)
    setTaskDraft(draft)
    setTaskSession((value) => value + 1)
    setView('task')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function openJob(value: Job) {
    setJob(value)
    if (value.request) {
      setTaskDraft(value.request)
      setTaskSession((current) => current + 1)
    }
    setView(value.status === 'complete' ? 'result' : 'progress')
  }

  function reuseJob(value: Job) {
    if (value.request) newTask(value.request)
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
      <Shell view={view} theme={theme} introStage={introStage} locatorBbox={locatorBbox} hasJob={Boolean(job)} hasResult={Boolean(job?.result)} onNavigate={setView} onNewTask={() => newTask()} onThemeChange={setTheme}>
        {view === 'models' && <ModelPromptView onUseTemplate={newTask} />}
        {view === 'data' && <DataCatalogView onCreateTask={() => setView('task')} />}
        <div hidden={view !== 'task'}>
          <TaskComposer key={taskSession} initialDraft={taskDraft} jobStatus={job?.status} onSubmit={submit} onBboxChange={setLocatorBbox} onOpenModels={() => setView('models')} onOpenData={() => setView('data')} />
        </div>
        {view === 'progress' && job && <ProgressView job={job} onJob={setJob} onComplete={completeJob} />}
        {view === 'result' && job && (
          <Suspense fallback={<div className="empty-state">正在加载交互式结果视图…</div>}>
            <ResultView job={job} theme={theme} />
          </Suspense>
        )}
        {view === 'archive' && <ArchiveView onOpen={openJob} onReuse={reuseJob} />}
      </Shell>
      {introStage !== 'complete' && (
        <IntroTransition theme={theme} bbox={locatorBbox} onSettling={settleIntro} onComplete={completeIntro} />
      )}
    </>
  )
}
