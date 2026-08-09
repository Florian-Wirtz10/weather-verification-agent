import { ArrowRight, Bot, Braces, CloudLightning, CopyCheck, Sparkles } from 'lucide-react'
import type { TaskPayload } from '../types'

interface PromptTemplate {
  title: string
  scope: string
  description: string
  draft: Partial<TaskPayload>
}

const templates: PromptTemplate[] = [
  {
    title: '单起报 · 双模型综合诊断',
    scope: '空间演变',
    description: '同步回波、评分曲线与多指标联合诊断。',
    draft: {
      case_id: '2025-04-11',
      models: ['StormDiT', '风雷'],
      scope: 'single',
      prompt: '使用10、20、30、40 dBZ阈值，查看6—180分钟StormDiT和风雷的预报—CREF同步回波动画以及CSI、POD、FAR曲线；再用20 dBZ把CSI、POD、FAR放在同一张图中比较，并结合各图说明回波位置、强度、命中和空报随时效的变化。',
    },
  },
  {
    title: '自定义多起报 · 同一有效时间',
    scope: '预报更新',
    description: '比较不同起报对同一时刻的预报修正。',
    draft: {
      case_id: '2026-04-08',
      models: ['StormDiT', '风雷'],
      scope: 'custom',
      prompt: '对比我选择的多个起报在页面所选同一有效时间的预报信息。StormDiT和风雷分别生成空间对比图，每个面板标注起报时间和预报时效，并共同对照同一张CREF观测；使用20 dBZ评分说明随着起报更新、时效缩短，回波位置、强度和命中空报发生了什么变化。',
    },
  },
  {
    title: '全部起报 · 稳定性评估',
    scope: '总体性能',
    description: '利用严格公共样本概括多起报统计规律。',
    draft: {
      case_id: '2025-04-11',
      models: ['StormDiT', '风雷'],
      scope: 'all',
      prompt: '使用10、20、30、40 dBZ阈值汇总全部完整起报，生成StormDiT与风雷的CSI、POD、FAR随时效曲线、多阈值评分分布、起报—时效分箱热力图和预报率—观测率图；重点比较两种模型在不同阈值和时效下的稳定性。',
    },
  },
  {
    title: 'Zero-shot Benchmark',
    scope: '基线对比',
    description: '在统一公共网格上比较多种零样本基线。',
    draft: {
      case_id: '2025-04-11',
      models: ['StormDiT', 'DiffCast', 'Persistence'],
      scope: 'all',
      bbox: [28.7, 32.51, 112.38, 116.19],
      prompt: '使用20、35、45 dBZ阈值，比较6—120分钟CSI、FAR、MAE、RMSE和5像元FSS，生成时效曲线、35 dBZ公共时效汇总图及CSI热力图，重点分析DiffCast和StormDiT相对持续性预报的有效提升。',
    },
  },
]

export function ModelPromptView({ onUseTemplate }: { onUseTemplate: (draft: Partial<TaskPayload>) => void }) {
  return (
    <div className="reference-page">
      <div className="page-heading">
        <div><h1>模型与提示词</h1><p>查看智能解析边界，并从经过验证的模板创建检验任务。</p></div>
      </div>

      <section className="model-role-grid">
        <article className="panel role-card primary-role">
          <div className="reference-icon"><Bot size={22} /></div>
          <span>语言模型</span><h2>DeepSeek · deepseek-chat</h2>
          <p>把自然语言转换为受约束的检验参数，并依据结构化指标生成AI解读；不生成或执行绘图代码。</p>
        </article>
        <article className="panel role-card">
          <div className="reference-icon"><CloudLightning size={22} /></div>
          <span>预报产品</span><h2>StormDiT · 风雷</h2>
          <p>支持单产品诊断与严格公共样本下的双模型公平对比。</p>
        </article>
        <article className="panel role-card">
          <div className="reference-icon"><Braces size={22} /></div>
          <span>比较基线</span><h2>Zero-shot Benchmark</h2>
          <p>统一网格和有效时效下比较 ConvLSTM、PredRNN、DiffCast 等基线。</p>
        </article>
      </section>

      <section className="panel template-library">
        <div className="panel-title"><span>推荐提示词模板</span><em>{String(templates.length).padStart(2, '0')}</em></div>
        <div className="template-grid">
          {templates.map((template) => (
            <article className="template-card" key={template.title}>
              <div><span>{template.scope}</span><Sparkles size={15} /></div>
              <h2>{template.title}</h2>
              <p>{template.description}</p>
              <blockquote>{template.draft.prompt}</blockquote>
              <button onClick={() => onUseTemplate(template.draft)}><CopyCheck size={15} /> 使用此模板 <ArrowRight size={15} /></button>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
