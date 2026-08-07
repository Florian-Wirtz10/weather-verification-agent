import { ArrowRight, Clock3 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Job } from '../types'

export function ArchiveView({ onOpen }: { onOpen: (job: Job) => void }) {
  const [jobs, setJobs] = useState<Job[]>([])
  useEffect(() => { api.jobs().then((value) => setJobs(value.jobs)).catch(() => setJobs([])) }, [])
  return (
    <div className="archive-page">
      <div className="page-heading"><div><h1>任务归档</h1><p>查看当前服务进程中的检验任务和结果。</p></div></div>
      <section className="panel archive-list">
        <div className="panel-title"><span>最近任务</span></div>
        {jobs.map((job) => <button key={job.id} onClick={() => onOpen(job)}><span className={`archive-state ${job.status}`} /><div><strong>{job.id.toUpperCase()}</strong><small><Clock3 size={12} /> {job.created_at} · {job.message}</small></div><em>{job.status === 'running' ? '执行中' : job.status === 'complete' ? '已完成' : job.status === 'failed' ? '失败' : '等待中'}</em><ArrowRight size={16} /></button>)}
        {!jobs.length && <div className="empty-state">当前服务进程还没有任务记录</div>}
      </section>
    </div>
  )
}
