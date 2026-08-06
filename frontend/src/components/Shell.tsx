import {
  Activity,
  Archive,
  BarChart3,
  BrainCircuit,
  FileChartColumn,
  Gauge,
  Layers3,
  Play,
  Radar,
  Settings,
} from 'lucide-react'
import type { PropsWithChildren } from 'react'
import type { ViewName } from '../types'

interface ShellProps extends PropsWithChildren {
  view: ViewName
  systemReady: boolean
  hasJob: boolean
  hasResult: boolean
  onNavigate: (view: ViewName) => void
}

const nav = [
  { view: 'task' as const, label: '检验任务', icon: Radar },
  { view: 'progress' as const, label: '执行状态', icon: Activity },
  { view: 'result' as const, label: '结果分析', icon: BarChart3 },
  { view: 'archive' as const, label: '任务归档', icon: Archive },
]

export function Shell({
  children,
  view,
  systemReady,
  hasJob,
  hasResult,
  onNavigate,
}: ShellProps) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <BrainCircuit size={22} />
          <span>强对流预报智能检验平台</span>
        </div>
        <nav className="topnav" aria-label="主导航">
          <button className={view === 'task' ? 'active' : ''} onClick={() => onNavigate('task')}>
            Verification
          </button>
          <button className={view === 'result' ? 'active' : ''} onClick={() => hasResult && onNavigate('result')}>
            Reports
          </button>
        </nav>
        <div className="system-state">
          <span className={`status-dot ${systemReady ? 'ready' : ''}`} />
          <span>{systemReady ? 'System Ready' : 'Connecting'}</span>
          <Settings size={18} />
        </div>
      </header>

      <aside className="sidebar">
        <div className="agent-identity">
          <div className="agent-mark"><Gauge size={22} /></div>
          <div>
            <strong>Convection Agent</strong>
            <span>EXPERT WORKSPACE</span>
          </div>
        </div>
        <button className="execute-button" onClick={() => onNavigate('task')}>
          <Play size={16} fill="currentColor" /> 新建检验任务
        </button>
        <div className="side-nav">
          {nav.map((item) => {
            const disabled = (item.view === 'progress' && !hasJob) || (item.view === 'result' && !hasResult)
            const Icon = item.icon
            return (
              <button
                key={item.view}
                className={view === item.view ? 'active' : ''}
                disabled={disabled}
                onClick={() => onNavigate(item.view)}
              >
                <Icon size={17} /> {item.label}
              </button>
            )
          })}
        </div>
        <div className="sidebar-meta">
          <div><Layers3 size={15} /> StormDiT · 风雷 · Baselines</div>
          <div><FileChartColumn size={15} /> CREF evidence contract</div>
        </div>
      </aside>

      <main className="workspace">{children}</main>
    </div>
  )
}
