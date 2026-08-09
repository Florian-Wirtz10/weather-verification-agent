import {
  Download,
  Expand,
  FileDown,
  X,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Artifact, Job, ThemeName } from '../types'
import { MetricChart } from './MetricChart'

function display(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3)
  if (Array.isArray(value)) return value.join(' · ')
  return String(value)
}

function GalleryImage({
  item,
  number,
  onExpand,
}: {
  item: Artifact
  number: number
  onExpand: (item: Artifact) => void
}) {
  return (
    <article className="gallery-card">
      <div className="gallery-image"><img src={item.url} alt={item.label} /><button onClick={() => onExpand(item)}><Expand size={16} /></button></div>
      <div><span className="figure-number">图 {number}</span><strong>{item.label}</strong><a href={item.url} download><Download size={14} /> {item.mime === 'image/gif' ? 'GIF' : 'PNG'}</a>{item.svg_url && <a href={item.svg_url} download>SVG</a>}</div>
    </article>
  )
}

export function ResultView({ job, theme }: { job: Job; theme: ThemeName }) {
  const result = job.result
  const [frameIndex, setFrameIndex] = useState(0)
  const [expanded, setExpanded] = useState<Artifact | null>(null)
  const [tableOpen, setTableOpen] = useState(false)
  const summaryColumns = useMemo(() => {
    const keys = new Set<string>()
    const summaryTable = result?.summary_table || []
    summaryTable.forEach((row) => Object.keys(row).forEach((key) => keys.add(key)))
    return [...keys]
  }, [result])

  if (!result) return <div className="empty-state">结果尚未生成</div>

  const frame = result.spatial_frames[frameIndex]
  const models = (result.context.model_labels || result.context.models || result.context.model_ids || []) as unknown[]
  const selectedLeads = Array.isArray(result.coverage.selected_leads)
    ? result.coverage.selected_leads.length
    : undefined
  const kindTitle = {
    single: '单一起报诊断结果', custom: '自定义多起报对比结果', all: '全部起报综合统计', benchmark: '多模型公共样本基准检验',
  }[result.kind]
  const kindLabel = {
    single: '单一起报', custom: '多起报对比', all: '综合统计', benchmark: '多模型基准检验',
  }[result.kind]

  return (
    <div className="result-page">
      <header className="result-header">
        <div>
          <h1>{kindTitle}</h1>
          <div className="result-context"><span>{models.map(display).join(' 与 ')}</span><span>严格匹配完成</span><span>任务 {job.id.toUpperCase()}</span></div>
        </div>
        <div className="download-actions">
          {result.downloads.map((item) => <a href={item.url} download key={item.name}><FileDown size={15} /> {item.label}</a>)}
        </div>
      </header>

      <div className="workflow-line"><span className="done">资料加载</span><i /><span className="done">指标计算</span><i /><span className="done">图表生成</span><i /><span className="done">结果解读</span></div>

      <div className="result-layout">
        <div className="result-main">
          <section className="stat-strip panel">
            <div><span>模型数量</span><strong>{models.length || 1}</strong></div>
            <div><span>起报样本</span><strong>{display(result.coverage.selected_initialization_count || result.coverage.sample_count || 1)}</strong></div>
            <div><span>严格匹配</span><strong>{display(result.coverage.selected_pairs || result.coverage.pair_count || selectedLeads)}</strong></div>
            <div><span>结果类型</span><strong>{kindLabel}</strong></div>
          </section>

          {frame && (
            <section className="panel spatial-panel">
              <div className="section-heading"><div><strong>空间分布时效浏览</strong></div><span>+{frame.lead_minutes} 分钟</span></div>
              <div className="spatial-image"><img src={frame.url} alt={frame.label} /><button onClick={() => setExpanded(frame)}><Expand size={18} /></button></div>
              <div className="time-slider"><span>+{result.spatial_frames[0].lead_minutes}m</span><input type="range" min="0" max={result.spatial_frames.length - 1} value={frameIndex} onChange={(event) => setFrameIndex(Number(event.target.value))} /><span>+{result.spatial_frames.at(-1)?.lead_minutes}m</span></div>
            </section>
          )}

          {result.curves.length > 0 && <section className="panel"><MetricChart curves={result.curves} theme={theme} /></section>}

          <section className="gallery-section">
            <div className="section-heading"><div><strong>科学图表</strong></div><span>{result.gallery.length} 幅</span></div>
            <div className="gallery-grid">{result.gallery.map((item, index) => <GalleryImage key={item.name} item={item} number={index + 1} onExpand={setExpanded} />)}</div>
          </section>

          {result.summary_table.length > 0 && (
            <section className="panel table-panel">
              <div className="section-heading"><div><strong>全局指标汇总</strong></div><button onClick={() => setTableOpen((value) => !value)}>{tableOpen ? '收起' : '查看全部'}</button></div>
              <div className={`scientific-table-wrap ${tableOpen ? 'open' : ''}`}>
                <table><thead><tr>{summaryColumns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{result.summary_table.map((row, index) => <tr key={index}>{summaryColumns.map((column) => <td key={column}>{display(row[column])}</td>)}</tr>)}</tbody></table>
              </div>
            </section>
          )}
        </div>

        <aside className="insight-panel panel">
          <div className="insight-heading"><div><strong>AI解读</strong></div></div>
          <div className="insight-content">
            <ReactMarkdown>{result.insight}</ReactMarkdown>
          </div>
          <div className="insight-contract">解读仅使用本次结构化评分事实，不读取或生成检验代码。</div>
        </aside>
      </div>

      {expanded && <div className="image-modal" onClick={() => setExpanded(null)}><div onClick={(event) => event.stopPropagation()}><button onClick={() => setExpanded(null)}><X size={20} /></button><img src={expanded.url} alt={expanded.label} /><span>{expanded.label}</span></div></div>}
    </div>
  )
}
