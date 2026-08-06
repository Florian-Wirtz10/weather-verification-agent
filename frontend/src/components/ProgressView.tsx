import { Check, Circle, Clipboard, Database, LoaderCircle, TerminalSquare, XCircle } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { Job, JobEvent } from '../types'

interface ProgressProps {
  job: Job
  onJob: (job: Job) => void
  onComplete: (job: Job) => void
}

const stages = [
  { id: 'parse', label: '解析与校验分析要求' },
  { id: 'align', label: '匹配预报、观测与时空网格' },
  { id: 'compute', label: '计算逐样本检验指标' },
  { id: 'render', label: '生成科学图表与交互资产' },
  { id: 'interpret', label: '生成证据约束型AI解读' },
]

export function ProgressView({ job, onJob, onComplete }: ProgressProps) {
  const [events, setEvents] = useState<JobEvent[]>([])
  const [copied, setCopied] = useState(false)
  const terminalRef = useRef<HTMLDivElement>(null)
  const activeIndex = Math.max(0, stages.findIndex((item) => item.id === job.stage))

  useEffect(() => {
    const source = new EventSource(`/api/jobs/${job.id}/events`)
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent
      setEvents((current) => current.some((item) => item.index === event.index) ? current : [...current, event])
    }
    source.addEventListener('end', () => source.close())
    const timer = window.setInterval(async () => {
      try {
        const latest = await api.job(job.id)
        onJob(latest)
        if (latest.status === 'complete' || latest.status === 'failed') {
          window.clearInterval(timer)
          source.close()
          if (latest.status === 'complete') onComplete(latest)
        }
      } catch {
        // A transient poll failure must not terminate the SSE progress stream.
      }
    }, 700)
    return () => {
      source.close()
      window.clearInterval(timer)
    }
  }, [job.id, onComplete, onJob])

  useEffect(() => {
    terminalRef.current?.scrollTo({ top: terminalRef.current.scrollHeight, behavior: 'smooth' })
  }, [events])

  const logText = useMemo(() => events.map((event) => `[${event.timestamp.slice(11, 19)}] ${event.stage.toUpperCase()}  ${event.message}`).join('\n'), [events])

  return (
    <div className="progress-page">
      <section className="pipeline-header panel">
        <div>
          <div className="eyebrow">PIPELINE EXECUTION</div>
          <h1>检验任务执行状态</h1>
          <p>{job.message}</p>
        </div>
        <div className="job-state"><span>JOB {job.id.toUpperCase()}</span><b className={job.status}>{job.status}</b></div>
      </section>

      <div className="progress-layout">
        <section className="panel stages-panel">
          <div className="panel-title"><Database size={18} /><span>Execution Stages</span></div>
          <div className="stage-list">
            {stages.map((stage, index) => {
              const complete = index < activeIndex || job.status === 'complete'
              const active = index === activeIndex && job.status === 'running'
              const failed = job.status === 'failed' && index === activeIndex
              return (
                <div className={`stage ${complete ? 'complete' : ''} ${active ? 'active' : ''} ${failed ? 'failed' : ''}`} key={stage.id}>
                  <div className="stage-node">
                    {complete ? <Check size={14} /> : failed ? <XCircle size={16} /> : active ? <LoaderCircle className="spin" size={15} /> : <Circle size={11} />}
                  </div>
                  <div><strong>{stage.label}</strong><span>{complete ? 'Completed' : active || failed ? job.message : 'Pending'}</span></div>
                </div>
              )
            })}
          </div>
          <div className="overall-progress"><div><span>总体进度</span><b>{Math.round(job.progress * 100)}%</b></div><div className="progress-track"><i style={{ width: `${job.progress * 100}%` }} /></div></div>
        </section>

        <section className="console-column">
          <div className="console panel">
            <div className="console-header"><div><TerminalSquare size={16} /> 实时执行日志</div><button onClick={async () => { await navigator.clipboard.writeText(logText); setCopied(true); window.setTimeout(() => setCopied(false), 1200) }}><Clipboard size={15} /> {copied ? '已复制' : '复制'}</button></div>
            <div className="console-body" ref={terminalRef}>
              {events.map((event) => <div key={event.index}><span>{event.timestamp.slice(11, 19)}</span><b>{event.stage.toUpperCase()}</b><em>{event.message}</em></div>)}
              {job.status === 'running' && <div className="console-cursor">▌</div>}
              {job.error && <div className="console-error">ERROR · {job.error}</div>}
            </div>
          </div>
          <div className={`integrity-card panel ${job.status}`}>
            <div className="integrity-icon"><Database size={24} /></div>
            <div><strong>{job.status === 'failed' ? 'Execution halted' : 'Data contract active'}</strong><span>{job.status === 'failed' ? job.error : '严格时间匹配、公共有效掩膜与确定性指标计算已启用。'}</span></div>
          </div>
        </section>
      </div>
    </div>
  )
}
