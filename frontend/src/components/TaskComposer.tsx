import {
  BookOpen,
  Check,
  LoaderCircle,
  Play,
  RotateCcw,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { CaseOption, Initialization, Scope, TaskPayload } from '../types'

interface TaskComposerProps {
  onSubmit: (payload: TaskPayload) => Promise<void>
  onBboxChange: (bbox: number[]) => void
  initialDraft?: Partial<TaskPayload>
  jobStatus?: 'queued' | 'running' | 'complete' | 'failed'
  onOpenModels: () => void
  onOpenData: () => void
}

const scopeOptions: { id: Scope; label: string; hint: string }[] = [
  { id: 'single', label: '单一起报', hint: '空间演变与逐时效诊断' },
  { id: 'custom', label: '自定义多起报', hint: '同一有效时间更新对比' },
  { id: 'all', label: '全部起报', hint: '综合统计与稳定性分析' },
]
const DEFAULT_CASE_ID = '2025-04-11'

export function TaskComposer({ onSubmit, onBboxChange, initialDraft, jobStatus, onOpenModels, onOpenData }: TaskComposerProps) {
  const [cases, setCases] = useState<CaseOption[]>([])
  const [caseId, setCaseId] = useState(initialDraft?.case_id || DEFAULT_CASE_ID)
  const [models, setModels] = useState<string[]>(initialDraft?.models || ['StormDiT', '风雷'])
  const [scope, setScope] = useState<Scope>(initialDraft?.scope || 'single')
  const [bbox, setBbox] = useState<number[]>(initialDraft?.bbox || [29, 33.5, 108, 116.5])
  const [completeOnly, setCompleteOnly] = useState(initialDraft?.complete_only ?? true)
  const [initializations, setInitializations] = useState<Initialization[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>(initialDraft?.initialization_ids || [])
  const [validTimes, setValidTimes] = useState<{ value: string; label: string }[]>([])
  const [validTime, setValidTime] = useState<string | null>(initialDraft?.comparison_valid_time || null)
  const [prompt, setPrompt] = useState(initialDraft?.prompt || '')
  const [defaultText, setDefaultText] = useState('')
  const [loadingCatalog, setLoadingCatalog] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const seededSelectionRef = useRef(initialDraft?.initialization_ids || [])
  const seededPromptRef = useRef(initialDraft?.prompt || '')
  const seededValidTimeRef = useRef(initialDraft?.comparison_valid_time || null)
  const initialCaseIdRef = useRef(caseId)
  const hasSeededBboxRef = useRef(Boolean(initialDraft?.bbox))

  const selectedCase = cases.find((item) => item.id === caseId)
  const benchmarkMode = models.some((model) => !['StormDiT', '风雷'].includes(model))
  const supplementalDomain = useMemo(() => {
    const constrainedInitializations = scope === 'all'
      ? initializations
      : initializations.filter((item) => selectedIds.includes(item.id))
    return constrainedInitializations.find((item) => item.domain)?.domain || null
  }, [initializations, scope, selectedIds])
  const effectiveBbox = useMemo(
    () => benchmarkMode
      ? [28.7, 32.51, 112.38, 116.19]
      : supplementalDomain || bbox,
    [benchmarkMode, bbox, supplementalDomain],
  )
  const bboxLocked = benchmarkMode || Boolean(supplementalDomain)
  const taskComplete = jobStatus === 'complete'
  const taskExecuting = submitting || jobStatus === 'queued' || jobStatus === 'running'

  useEffect(() => {
    if (jobStatus === 'complete' || jobStatus === 'failed') setSubmitting(false)
  }, [jobStatus])

  useEffect(() => {
    onBboxChange(effectiveBbox)
  }, [effectiveBbox, onBboxChange])

  useEffect(() => {
    api.options()
      .then(({ cases: values }) => {
        setCases(values)
        const selected = values.find((item) => item.id === initialCaseIdRef.current)
        const fallback = values.find((item) => item.id === DEFAULT_CASE_ID) || values[0]
        if (!selected && fallback) setCaseId(fallback.id)
        if (!hasSeededBboxRef.current && (selected || fallback)) setBbox((selected || fallback).default_bbox)
      })
      .catch((reason) => setError(reason.message))
  }, [])

  useEffect(() => {
    if (!models.length) return
    setLoadingCatalog(true)
    setError('')
    api.initializations(caseId, models, completeOnly)
      .then(({ initializations: values }) => {
        setInitializations(values)
        const availableIds = new Set(values.map((item) => item.id))
        const seeded = seededSelectionRef.current.filter((id) => availableIds.has(id))
        seededSelectionRef.current = []
        if (seeded.length) setSelectedIds(scope === 'single' ? seeded.slice(0, 1) : seeded.slice(0, 8))
        else if (scope === 'single') setSelectedIds(values.slice(0, 1).map((item) => item.id))
        else if (scope === 'custom') setSelectedIds(values.slice(0, 3).map((item) => item.id))
        else setSelectedIds([])
      })
      .catch((reason) => {
        setInitializations([])
        setError(reason.message)
      })
      .finally(() => setLoadingCatalog(false))
  }, [caseId, models, completeOnly, scope])

  useEffect(() => {
    if (!models.length) return
    api.defaultPrompt(caseId, models, scope, completeOnly)
      .then(({ prompt: value }) => {
        setDefaultText(value)
        if (seededPromptRef.current) {
          setPrompt(seededPromptRef.current)
          seededPromptRef.current = ''
        } else {
          setPrompt(value)
        }
      })
      .catch((reason) => setError(reason.message))
  }, [caseId, models, scope, completeOnly])

  useEffect(() => {
    if (scope !== 'custom' || selectedIds.length < 2 || benchmarkMode) {
      setValidTimes([])
      setValidTime(null)
      return
    }
    api.commonTimes(caseId, models, selectedIds, completeOnly)
      .then(({ valid_times }) => {
        setValidTimes(valid_times)
        const seeded = seededValidTimeRef.current
        setValidTime(
          (seeded && valid_times.some((item) => item.value === seeded) ? seeded : null)
          || valid_times[Math.floor(valid_times.length / 2)]?.value
          || null,
        )
        seededValidTimeRef.current = null
      })
      .catch((reason) => setError(reason.message))
  }, [caseId, models, scope, selectedIds, completeOnly, benchmarkMode])

  const plan = useMemo(() => {
    const selected = initializations.filter((item) => selectedIds.includes(item.id))
    const leads = selected.flatMap((item) => item.available_leads)
    const minLead = leads.length ? Math.min(...leads) : 6
    const maxLead = leads.length ? Math.max(...leads) : 180
    return [
      { label: '目标模型', value: models.join(' · ') || '未选择' },
      { label: '观测基准', value: 'CREF 雷达组合反射率' },
      { label: '样本组织', value: scopeOptions.find((item) => item.id === scope)?.label || scope },
      { label: '可用时效', value: `+${minLead}—+${maxLead} 分钟` },
    ]
  }, [initializations, selectedIds, models, scope])

  function changeCase(value: string) {
    const next = cases.find((item) => item.id === value)
    setCaseId(value)
    setModels(['StormDiT', '风雷'].filter((model) => next?.models.includes(model)))
    if (next) setBbox(next.default_bbox)
  }

  function toggleModel(model: string) {
    setModels((current) => {
      return current.includes(model)
        ? current.filter((item) => item !== model)
        : [...current, model]
    })
  }

  function toggleInitialization(id: string) {
    if (scope === 'single') {
      setSelectedIds([id])
      return
    }
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(0, 8),
    )
  }

  async function submit() {
    setError('')
    if (!models.length) return setError('请至少选择一个预报模型。')
    if (scope === 'single' && selectedIds.length !== 1) return setError('请选择一个起报。')
    if (scope === 'custom' && (selectedIds.length < 2 || selectedIds.length > 8)) {
      return setError('请选择2—8个起报。')
    }
    setSubmitting(true)
    try {
      await onSubmit({
        case_id: caseId,
        models,
        scope,
        bbox: effectiveBbox,
        complete_only: completeOnly,
        initialization_ids: selectedIds,
        comparison_valid_time: validTime,
        prompt,
        use_default_plan: prompt.trim() === defaultText.trim(),
      })
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '任务提交失败')
      setSubmitting(false)
    }
  }

  return (
    <div className="task-page">
      <div className="page-heading">
        <div>
          <h1>创建强对流检验任务</h1>
          <p>选择个例、模型和起报时次，配置本次检验范围。</p>
        </div>
      </div>

      <div className="task-grid">
        <section className="panel configuration-panel">
          <div className="panel-title"><span>检验配置</span><button className="panel-link" onClick={onOpenData}><BookOpen size={13} /> 了解资料</button><em>01</em></div>
          <label className="field-label">检验个例</label>
          <select value={caseId} disabled={taskComplete} onChange={(event) => changeCase(event.target.value)}>
            {cases.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
          </select>

          <label className="field-label">预报模型</label>
          {(['case', 'benchmark'] as const).map((family) => {
            const values = selectedCase?.model_groups[family] || []
            if (!values.length) return null
            return <div className="model-family" key={family}>
              <span>{family === 'case' ? '业务模型' : '基线模型'}</span>
              <div className="model-grid">
                {values.map((model) => (
                  <button
                    key={model}
                    className={`model-chip ${models.includes(model) ? 'selected' : ''}`}
                    disabled={taskComplete}
                    onClick={() => toggleModel(model)}
                  >
                    {models.includes(model) && <Check size={13} />} {model}
                  </button>
                ))}
              </div>
            </div>
          })}

          <label className="field-label">检验范围</label>
          <div className="scope-grid">
            {scopeOptions.map((item) => (
              <button key={item.id} disabled={taskComplete} className={scope === item.id ? 'selected' : ''} onClick={() => setScope(item.id)}>
                <strong>{item.label}</strong><span>{item.hint}</span>
              </button>
            ))}
          </div>

          <div className="field-row bbox-row">
            {['南界 °N', '北界 °N', '西界 °E', '东界 °E'].map((label, index) => (
              <label key={label}>{label}<input type="number" step="0.1" disabled={taskComplete || bboxLocked} value={effectiveBbox[index]} onChange={(event) => setBbox((current) => current.map((value, i) => i === index ? Number(event.target.value) : value))} /></label>
            ))}
          </div>
          <label className="check-row">
            <input type="checkbox" disabled={taskComplete} checked={completeOnly} onChange={(event) => setCompleteOnly(event.target.checked)} />
            仅使用资料完整的起报
          </label>
        </section>

        <section className="panel sample-panel">
          <div className="panel-title"><span>起报样本</span><em>02</em></div>
          {loadingCatalog ? (
            <div className="loading-block"><LoaderCircle className="spin" /> 正在建立资料索引…</div>
          ) : (
            <div className="initialization-list">
              {scope === 'all' && <div className="all-samples"><strong>全部 {initializations.length} 个可用起报</strong><span>将汇总所有严格匹配样本</span></div>}
              {scope !== 'all' && initializations.map((item) => (
                <button key={item.id} disabled={taskComplete} className={selectedIds.includes(item.id) ? 'selected' : ''} onClick={() => toggleInitialization(item.id)}>
                  <span className="select-indicator">{selectedIds.includes(item.id) ? <Check size={13} /> : null}</span>
                  <span><strong>{item.label}</strong><small>+{item.available_leads[0]}—+{item.available_leads.at(-1)} 分钟 · {item.complete ? '完整' : '部分'}</small></span>
                </button>
              ))}
              {!initializations.length && <div className="empty-state">当前组合没有可检验起报</div>}
            </div>
          )}
          {scope === 'custom' && validTimes.length > 0 && (
            <div className="valid-time-select">
              <label>共同有效时间</label>
              <select value={validTime || ''} disabled={taskComplete} onChange={(event) => setValidTime(event.target.value)}>
                {validTimes.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
              </select>
            </div>
          )}
        </section>
      </div>

      <section className="panel agent-panel">
        <div className="agent-input-header"><div><span>检验要求</span><button className="inline-reference" onClick={onOpenModels}><BookOpen size={13} /> 查看模型与提示词</button></div><button disabled={taskComplete} onClick={() => setPrompt(defaultText)}><RotateCcw size={14} /> 恢复推荐要求</button></div>
        <textarea value={prompt} disabled={taskComplete} onChange={(event) => setPrompt(event.target.value)} placeholder="请输入检验分析需求…" />
        <div className="agent-action"><span>已输入 {prompt.length} 个字符</span></div>
        <div className="parsed-plan">
          <div className="plan-entities">
            <h3>规范化任务参数</h3>
            {plan.map((item) => <div key={item.label}><span>{item.label}</span><strong>{item.value}</strong></div>)}
          </div>
          <div className="plan-stages">
            <h3>确定性执行流程</h3>
            {['严格时空匹配', '阈值事件提取', '列联表与连续量计算', '科学图表与证据解读'].map((item, index) => (
              <div key={item}><b>{index + 1}</b><span>{item}</span></div>
            ))}
          </div>
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="execute-row">
          <button className={`primary-action ${taskComplete ? 'task-complete' : ''}`} disabled={taskExecuting || taskComplete || loadingCatalog} onClick={submit}>
            {taskComplete ? <Check size={17} /> : taskExecuting ? <LoaderCircle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}
            {taskComplete ? '任务已完成' : taskExecuting ? '正在执行检验' : '执行检验任务'}
          </button>
        </div>
      </section>
    </div>
  )
}
