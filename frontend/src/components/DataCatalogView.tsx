import { Clock3, Database, Grid3X3, MapPinned, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { DataMetadata } from '../types'

export function DataCatalogView({ onCreateTask }: { onCreateTask: () => void }) {
  const [metadata, setMetadata] = useState<DataMetadata | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.metadata().then(setMetadata).catch((reason) => setError(reason.message))
  }, [])

  return (
    <div className="reference-page">
      <div className="page-heading reference-heading">
        <div><h1>数据资料</h1><p>当前接入资料的时空口径与严格匹配规则。</p></div>
        <button className="primary-action compact-action" onClick={onCreateTask}>返回检验配置</button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {!metadata && !error && <div className="empty-state">正在读取资料契约…</div>}
      {metadata && (
        <>
          <section className="data-product-grid">
            {metadata.products.map((product) => (
              <article className="panel data-product-card" key={product.name}>
                <div className="data-product-title"><div className="reference-icon"><Database size={19} /></div><div><span>{product.kind}</span><h2>{product.name}</h2></div><em>{product.format}</em></div>
                <dl>
                  <div><dt><Grid3X3 size={14} /> 原生网格</dt><dd>{product.grid}</dd></div>
                  <div><dt><Clock3 size={14} /> 时间口径</dt><dd>{product.clock}</dd></div>
                  <div><dt><Clock3 size={14} /> 时间分辨率</dt><dd>{product.cadence}</dd></div>
                  <div><dt><MapPinned size={14} /> 空间范围</dt><dd>{product.domain}</dd></div>
                </dl>
              </article>
            ))}
          </section>

          <div className="data-lower-grid">
            <section className="panel case-catalog">
              <div className="panel-title"><span>已接入个例</span></div>
              {metadata.cases.map((item) => <div className="case-row" key={item.id}><time>{item.id}</time><div><strong>{item.label}</strong><span>{item.availability}</span></div></div>)}
            </section>
            <section className="panel matching-rules">
              <div className="panel-title"><span>公平检验规则</span></div>
              {metadata.rules.map((rule, index) => <div key={rule}><ShieldCheck size={17} /><span><b>{String(index + 1).padStart(2, '0')}</b>{rule}</span></div>)}
            </section>
          </div>
        </>
      )}
    </div>
  )
}
