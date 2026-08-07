import {
  Activity,
  Archive,
  BarChart3,
  CloudSun,
  Moon,
  Play,
  Radar,
  Sun,
} from 'lucide-react'
import { lazy, Suspense, type PropsWithChildren } from 'react'
import type { ThemeName, ViewName } from '../types'

const ChinaLocatorMap = lazy(() => import('./ChinaLocatorMap').then((module) => ({ default: module.ChinaLocatorMap })))

interface ShellProps extends PropsWithChildren {
  view: ViewName
  theme: ThemeName
  locatorBbox: number[]
  hasJob: boolean
  hasResult: boolean
  onNavigate: (view: ViewName) => void
  onThemeChange: (theme: ThemeName) => void
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
  theme,
  locatorBbox,
  hasJob,
  hasResult,
  onNavigate,
  onThemeChange,
}: ShellProps) {
  return (
    <div className={`app-shell theme-${theme}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true"><CloudSun size={20} /></div>
          <strong>强对流预报智能检验平台</strong>
        </div>
        <nav className="topnav" aria-label="主导航">
          <button className={view === 'task' ? 'active' : ''} onClick={() => onNavigate('task')}>
            检验工作台
          </button>
          <button className={view === 'result' ? 'active' : ''} onClick={() => hasResult && onNavigate('result')}>
            分析报告
          </button>
        </nav>
        <button
          className="theme-toggle"
          type="button"
          aria-label={theme === 'dark' ? '切换到明亮界面' : '切换到深色界面'}
          title={theme === 'dark' ? '切换到明亮界面' : '切换到深色界面'}
          aria-pressed={theme === 'dark'}
          onClick={() => onThemeChange(theme === 'dark' ? 'light' : 'dark')}
        >
          <span className="theme-toggle-icon" key={theme}>{theme === 'dark' ? <Sun size={19} /> : <Moon size={19} />}</span>
        </button>
      </header>

      <aside className="sidebar">
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
        <Suspense fallback={<div className="locator-map locator-map-loading" aria-hidden="true" />}>
          <ChinaLocatorMap bbox={locatorBbox} theme={theme} />
        </Suspense>
        <div className="sidebar-meta">
          <strong>当前资料</strong>
          <span>StormDiT、风雷及基线模型</span>
          <span>CREF 雷达组合反射率</span>
        </div>
      </aside>

      <main className="workspace">{children}</main>
    </div>
  )
}
