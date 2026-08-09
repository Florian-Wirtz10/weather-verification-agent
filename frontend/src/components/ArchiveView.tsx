import { ArrowRight, Clock3, RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Job } from '../types'

interface ArchiveProps {
  onOpen: (job: Job) => void
  onReuse: (job: Job) => void
}

function createdLabel(value: string) {
  return new Date(value).toLocaleString()
}

const scopeLabels = { single: '单一起报', custom: '自定义多起报', all: '全部起报' }

function requestLabel(job: Job) {
  if (!job.request) return '历史任务'
  return [
    job.request.case_id,
    job.request.models.join(' · '),
    scopeLabels[job.request.scope],
  ].join('｜')
}

export function ArchiveView({ onOpen, onReuse }: ArchiveProps) {
  const [jobs, setJobs] = useState<Job[]>([])
  useEffect(() => { api.jobs().then((value) => setJobs(value.jobs)).catch(() => setJobs([])) }, [])
  return (
    <div className="archive-page">
      <div className="page-heading"><div><h1>任务归档</h1><p>查看当前服务进程中的检验任务和结果。</p></div></div>
      <section className="panel archive-list">
        <div className="panel-title"><span>最近任务</span></div>
        {jobs.map((job) => <article className="archive-item" key={job.id}>
          <span className={`archive-state ${job.status}`} />
          <button className="archive-open" onClick={() => onOpen(job)}>
            <div><strong>{job.id.toUpperCase()}</strong><small><Clock3 size={12} /> {createdLabel(job.created_at)}</small><small className="archive-request">{requestLabel(job)}</small></div>
          </button>
          <em>{job.status === 'running' ? '执行中' : job.status === 'complete' ? '已完成' : job.status === 'failed' ? '失败' : '等待中'}</em>
          <button className="archive-reuse" disabled={!job.request} onClick={() => onReuse(job)} title="使用相同配置新建任务"><RotateCcw size={14} /> 复用配置</button>
          <button className="archive-arrow" onClick={() => onOpen(job)} aria-label="打开任务"><ArrowRight size={16} /></button>
        </article>)}
        {!jobs.length && <div className="empty-state">当前服务进程还没有任务记录</div>}
      </section>
    </div>
  )
}
