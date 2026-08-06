import { Archive, ArrowRight, Clock3 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Job } from '../types'

export function ArchiveView({ onOpen }: { onOpen: (job: Job) => void }) {
  const [jobs, setJobs] = useState<Job[]>([])
  useEffect(() => { api.jobs().then((value) => setJobs(value.jobs)).catch(() => setJobs([])) }, [])
  return (
    <div className="archive-page">
      <div className="page-heading"><div><div className="eyebrow">SESSION ARCHIVE</div><h1>本次服务任务归档</h1><p>任务记录保存在当前服务进程中，结果资产写入本地 outputs。</p></div></div>
      <section className="panel archive-list">
        <div className="panel-title"><Archive size={18} /><span>Recent Jobs</span></div>
        {jobs.map((job) => <button key={job.id} onClick={() => onOpen(job)}><span className={`archive-state ${job.status}`} /><div><strong>{job.id.toUpperCase()}</strong><small><Clock3 size={12} /> {job.created_at} · {job.message}</small></div><em>{job.status}</em><ArrowRight size={16} /></button>)}
        {!jobs.length && <div className="empty-state">当前服务进程还没有任务记录</div>}
      </section>
    </div>
  )
}
