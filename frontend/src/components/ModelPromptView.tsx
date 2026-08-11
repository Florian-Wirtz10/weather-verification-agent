import { ArrowRight, Check, Copy, Image as ImageIcon } from 'lucide-react'
import { useState } from 'react'
import type { Scope, TaskPayload } from '../types'
import { ChartPreview, type ChartPreviewKind } from './ChartPreview'

type GuideId = 'single' | 'custom' | 'all' | 'benchmark'

interface FigureCapability {
  name: string
  purpose: string
  controls: string
  preview: ChartPreviewKind
}

interface PromptExample {
  title: string
  note: string
  draft: Partial<TaskPayload>
}

interface ResultGuide {
  id: GuideId
  number: string
  label: string
  shortLabel: string
  summary: string
  scope: Scope
  sampleRule: string
  figures: FigureCapability[]
  prompts: PromptExample[]
}

const guides: ResultGuide[] = [
  {
    id: 'single',
    number: '01',
    label: '单一起报诊断',
    shortLabel: '单一起报',
    summary: '观察一次起报从短时效到长时效的空间演变和评分变化。',
    scope: 'single',
    sampleRule: '1 个起报；业务模型可选 StormDiT、风雷或二者公共样本',
    figures: [
      { name: '同步回波动画', purpose: '逐时效对照预报与 CREF 空间演变', controls: '时效范围、抽帧步长、1—4 fps', preview: 'animation' },
      { name: '指定时效空间对比图', purpose: '检查某一时效的回波位置、形态和强度', controls: '指定 6 分钟倍数时效', preview: 'spatial' },
      { name: '单指标多阈值曲线', purpose: '比较 CSI、POD、FAR 或 Bias 随时效变化', controls: '指标、阈值、时效范围', preview: 'line' },
      { name: '固定阈值多指标曲线', purpose: '在一张图中联合查看 CSI、POD 与 FAR', controls: '诊断阈值、时效范围', preview: 'multi-line' },
    ],
    prompts: [
      {
        title: '完整时空演变',
        note: '动画与指标曲线配合，适合现场演示',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', '风雷'], scope: 'single',
          prompt: '使用10、20、30、40 dBZ阈值，查看6—180分钟StormDiT和风雷的预报—CREF同步回波动画以及CSI、POD、FAR曲线；再用20 dBZ把CSI、POD、FAR放在同一张图中比较，并结合各图说明回波位置、强度、命中和空报随时效的变化。',
        },
      },
      {
        title: '指定时效精查',
        note: '集中检查一个关键时刻的空间误差',
        draft: {
          case_id: '2026-04-08', models: ['StormDiT', '风雷'], scope: 'single',
          prompt: '生成+60分钟StormDiT、风雷与CREF的静态空间对比图，使用20和35 dBZ阈值绘制CSI、POD、FAR时效曲线；重点判断强回波位置偏差、漏报和空报分别出现在哪里。',
        },
      },
      {
        title: 'StormDiT 单产品诊断',
        note: '保留弱回波，观察模型自身演变',
        draft: {
          case_id: '2026-04-08', models: ['StormDiT'], scope: 'single',
          prompt: '仅检验StormDiT，生成6—120分钟预报—CREF同步回波动画，绘制10、20、30、40 dBZ下的CSI和Bias时效曲线，并用20 dBZ联合展示CSI、POD、FAR。',
        },
      },
    ],
  },
  {
    id: 'custom',
    number: '02',
    label: '自定义多起报对比',
    shortLabel: '自定义多起报',
    summary: '用少量指定起报研究预报更新，或形成具有明确范围的样本子集。',
    scope: 'custom',
    sampleRule: '2—8 个起报；可选择这些起报共同拥有的一个有效时间',
    figures: [
      { name: '同一有效时间空间对比图', purpose: '比较不同起报对同一时刻的预报更新', controls: '起报子集、共同有效时间', preview: 'spatial' },
      { name: '子集汇总时效曲线', purpose: '比较所选起报内的平均 CSI、POD、FAR 或 Bias', controls: '指标、阈值、时效范围', preview: 'line' },
      { name: '起报—时效评分矩阵', purpose: '定位特定起报或时效的评分异常', controls: '指标、诊断阈值', preview: 'heatmap' },
      { name: '评分分布图', purpose: '查看多个起报和多个阈值的离散程度', controls: '指标、阈值集合', preview: 'distribution' },
      { name: '预报率—观测率散点图', purpose: '判断所选子集整体偏报、漏报或空报倾向', controls: '阈值集合', preview: 'scatter' },
      { name: '变量相关性热力图', purpose: '检查强度、误差和位置诊断量与评分的关系', controls: 'Pearson / Spearman、阈值', preview: 'correlation' },
    ],
    prompts: [
      {
        title: '同一有效时间更新对比',
        note: '自定义多起报最具代表性的空间诊断',
        draft: {
          case_id: '2026-04-08', models: ['StormDiT', '风雷'], scope: 'custom',
          prompt: '对比我选择的多个起报在页面所选同一有效时间的预报信息。StormDiT和风雷分别生成空间对比图，每个面板标注起报时间和预报时效，并共同对照同一张CREF观测；使用20 dBZ评分说明随着起报更新、时效缩短，回波位置、强度和命中空报发生了什么变化。',
        },
      },
      {
        title: '少量样本综合比较',
        note: '兼顾曲线、矩阵与离散样本分布',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', '风雷'], scope: 'custom',
          prompt: '对所选起报使用10、20、30、40 dBZ阈值，生成CSI、POD、FAR汇总时效曲线、20 dBZ CSI起报—时效矩阵和多阈值评分分布图，比较两个模型在这个起报子集中的稳定性。',
        },
      },
      {
        title: '偏报与物理诊断',
        note: '侧重预报率及诊断变量之间的关系',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', '风雷'], scope: 'custom',
          prompt: '使用20 dBZ分析我选择的多个起报，生成预报率—观测率散点图、Spearman变量相关性热力图，并绘制Bias随时效变化曲线；说明两个模型的偏报方向及其与回波强度、位置误差的关系。',
        },
      },
    ],
  },
  {
    id: 'all',
    number: '03',
    label: '全部起报综合统计',
    shortLabel: '全部起报',
    summary: '汇总当前个例内所有符合完整性条件的起报，评估总体性能和稳定性。',
    scope: 'all',
    sampleRule: '自动使用全部可用起报；双模型模式使用三方严格公共样本',
    figures: [
      { name: '全部起报汇总时效曲线', purpose: '观察总体评分随预报时效的变化', controls: '指标、阈值、时效范围', preview: 'line' },
      { name: '固定阈值多指标曲线', purpose: '同图比较 CSI、POD 与 FAR', controls: '诊断阈值', preview: 'multi-line' },
      { name: '起报—时效评分矩阵', purpose: '定位不同起报与预报时效的稳定结构和异常', controls: '指标、诊断阈值', preview: 'heatmap' },
      { name: '多阈值评分分布图', purpose: '比较模型在不同回波量级下的评分分布', controls: '指标、阈值集合', preview: 'distribution' },
      { name: '预报率—观测率散点图', purpose: '检查大样本下的系统性偏报方向', controls: '阈值集合', preview: 'scatter' },
      { name: '变量相关性热力图', purpose: '展示诊断变量间的相关结构', controls: 'Pearson / Spearman、阈值', preview: 'correlation' },
    ],
    prompts: [
      {
        title: '全部起报完整分析',
        note: '覆盖当前全部起报统计图',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', '风雷'], scope: 'all',
          prompt: '使用10、20、30、40 dBZ阈值汇总全部完整起报，生成StormDiT与风雷的CSI、POD、FAR随时效曲线、20 dBZ多指标联合曲线、多阈值评分分布、CSI起报—时效评分矩阵、Spearman变量相关性热力图和预报率—观测率散点图。',
        },
      },
      {
        title: '强回波能力评估',
        note: '减少图量，集中检查30 dBZ以上回波',
        draft: {
          case_id: '2026-04-08', models: ['StormDiT', '风雷'], scope: 'all',
          prompt: '仅分析30和40 dBZ，生成6—180分钟CSI、POD、FAR和Bias时效曲线、40 dBZ CSI起报—时效热力图及评分分布图，重点说明强回波命中能力和空报风险。',
        },
      },
      {
        title: '偏差与可解释性分析',
        note: '侧重预报率、位置误差与评分的联系',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', '风雷'], scope: 'all',
          prompt: '使用20 dBZ生成Bias时效曲线、预报率—观测率散点图和Spearman变量相关性热力图，结合CSI、POD、FAR说明两个模型的偏报方向、位置误差及回波强度误差。',
        },
      },
    ],
  },
  {
    id: 'benchmark',
    number: '04',
    label: 'Zero-shot Benchmark',
    shortLabel: '多模型基准',
    summary: '在武汉公共样本网格和共同可用时效内比较 StormDiT 与零样本基线。',
    scope: 'all',
    sampleRule: '模型共同最大时效由所选模型中最短的原生预报长度决定',
    figures: [
      { name: '多模型多阈值评分曲线', purpose: '比较 CSI、POD、FAR、Bias、MAE 或 RMSE', controls: '指标、阈值、公共时效', preview: 'line' },
      { name: 'FSS空间尺度曲线', purpose: '比较不同邻域容忍度下的空间技巧', controls: '1、3、5 像元邻域', preview: 'fss' },
      { name: '公共时段汇总比较图', purpose: '对所选共同时间段作多指标横向比较', controls: '指标、诊断阈值', preview: 'summary' },
      { name: '模型—时效评分热力图', purpose: '同时查看多个模型在各预报时效的差异', controls: '指标、诊断阈值', preview: 'heatmap' },
      { name: '多模型空间对比图', purpose: '比较单一起报、单个时效的空间预报', controls: '仅单一起报；指定时效', preview: 'spatial' },
    ],
    prompts: [
      {
        title: '三模型综合基准',
        note: '在较长共同预报窗口内进行代表性比较',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', 'DiffCast', 'Persistence'], scope: 'all', bbox: [28.7, 32.51, 112.38, 116.19],
          prompt: '使用20、35、45 dBZ阈值，比较6—120分钟CSI、FAR、MAE、RMSE和5像元FSS，生成时效曲线、35 dBZ公共时效汇总图及CSI热力图，重点分析DiffCast和StormDiT相对持续性预报的有效提升。',
        },
      },
      {
        title: '多基线分类评分',
        note: '比较较多模型，公共时效自动收缩',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', 'ConvLSTM', 'PredRNN', 'Earthformer', 'Persistence'], scope: 'all', bbox: [28.7, 32.51, 112.38, 116.19],
          prompt: '使用20、35、45 dBZ阈值，在所有所选模型的严格公共时效内生成CSI、POD、FAR曲线、35 dBZ公共时段汇总比较图和CSI模型—时效热力图。',
        },
      },
      {
        title: 'FSS空间尺度比较',
        note: '重点观察空间容忍度对模型排名的影响',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', 'DiffCast', 'NowCastNet', 'Persistence'], scope: 'all', bbox: [28.7, 32.51, 112.38, 116.19],
          prompt: '使用20和35 dBZ阈值，绘制1、3、5像元邻域FSS随时效变化曲线，并生成35 dBZ公共时段FSS汇总比较图和模型—时效热力图，说明邻域尺度改变后模型差异如何变化。',
        },
      },
      {
        title: '单起报空间基准',
        note: '同一时效直接比较多个基线的空间预报',
        draft: {
          case_id: '2025-04-11', models: ['StormDiT', 'DiffCast', 'Persistence'], scope: 'single', bbox: [28.7, 32.51, 112.38, 116.19],
          prompt: '生成+60分钟StormDiT、DiffCast、Persistence与CREF的空间对比图，同时使用20和35 dBZ绘制CSI、POD、FAR及5像元FSS时效曲线。',
        },
      },
    ],
  },
]

export function ModelPromptView({ onUseTemplate }: { onUseTemplate: (draft: Partial<TaskPayload>) => void }) {
  const [selectedId, setSelectedId] = useState<GuideId>('single')
  const [copiedTitle, setCopiedTitle] = useState('')
  const guide = guides.find((item) => item.id === selectedId) || guides[0]

  async function copyPrompt(example: PromptExample) {
    await navigator.clipboard.writeText(example.draft.prompt || '')
    setCopiedTitle(example.title)
    window.setTimeout(() => setCopiedTitle(''), 1200)
  }

  return (
    <div className="reference-page prompt-guide-page">
      <div className="page-heading prompt-guide-heading">
        <div><h1>检验功能与提示词</h1><p>先选择结果类型，再查看可生成图表和可直接使用的检验要求。</p></div>
      </div>

      <section className="panel prompt-guide-shell">
        <aside className="result-type-index" aria-label="结果类型">
          <div className="index-heading"><span>检验类型</span></div>
          {guides.map((item) => (
            <button key={item.id} className={item.id === selectedId ? 'active' : ''} onClick={() => setSelectedId(item.id)}>
              <div><strong>{item.shortLabel}</strong><small>支持 {item.figures.length} 类图表</small></div><ArrowRight size={15} />
            </button>
          ))}
        </aside>

        <article className="prompt-guide-content">
          <header className="result-guide-header">
            <h2>{guide.label}</h2>
            <p>{guide.summary}</p>
            <dl><div><dt>样本范围</dt><dd>{guide.sampleRule}</dd></div><div><dt>页面选择</dt><dd>{guide.shortLabel}</dd></div></dl>
          </header>

          <section className="guide-section capability-section">
            <div className="guide-section-heading"><div><h3>可生成图表</h3></div><em>共 {guide.figures.length} 类</em></div>
            <div className="capability-table" role="table">
              <div className="capability-row capability-head" role="row"><span>图表</span><span>主要用途</span><span>可控内容</span></div>
              {guide.figures.map((figure) => <div className="capability-row" role="row" key={figure.name}>
                <div className="capability-preview-trigger" tabIndex={0} aria-label={`${figure.name}，悬停或聚焦查看示意图`}>
                  <strong>{figure.name}</strong><ImageIcon size={13} aria-hidden="true" />
                  <div className="chart-preview-popover" role="tooltip">
                    <div className="chart-preview-title"><strong>{figure.name}</strong><span>形式示意</span></div>
                    <ChartPreview kind={figure.preview} />
                    <p>实际图表由所选资料、模型、阈值和时效计算生成。</p>
                  </div>
                </div>
                <span>{figure.purpose}</span><small>{figure.controls}</small>
              </div>)}
            </div>
          </section>

          <section className="guide-section prompt-examples-section">
            <div className="guide-section-heading"><div><h3>提示词示例</h3></div><em>共 {guide.prompts.length} 条</em></div>
            <div className="prompt-example-list">
              {guide.prompts.map((example) => (
                <article className="prompt-example" key={example.title}>
                  <div className="prompt-example-body"><header><strong>{example.title}</strong><span>{example.note}</span></header><p>{example.draft.prompt}</p></div>
                  <div className="prompt-example-actions">
                    <button onClick={() => copyPrompt(example)}>{copiedTitle === example.title ? <Check size={14} /> : <Copy size={14} />}{copiedTitle === example.title ? '已复制' : '复制'}</button>
                    <button className="use-prompt" onClick={() => onUseTemplate(example.draft)}>填入工作台 <ArrowRight size={14} /></button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </article>
      </section>
    </div>
  )
}
