import { Pause, Play } from 'lucide-react'
import { useState } from 'react'

export type ChartPreviewKind =
  | 'animation'
  | 'spatial'
  | 'line'
  | 'multi-line'
  | 'heatmap'
  | 'distribution'
  | 'scatter'
  | 'correlation'
  | 'fss'
  | 'summary'

function Axes() {
  return <>
    <path d="M38 18V142H292" className="preview-axis" />
    <path d="M38 111H292M38 79H292M38 47H292" className="preview-guide" />
  </>
}

function SpatialPanels({ animated = false }: { animated?: boolean }) {
  return <>
    {[15, 116, 217].map((x, index) => <g key={x} transform={`translate(${x} 31)`}>
      <rect width="88" height="104" rx="3" className="preview-map" />
      <g className={animated ? `preview-echo preview-echo-${index + 1}` : undefined}>
        <path d="M8 77c11-21 17-35 31-40 12-4 13 7 21 7 9 0 11-9 19-3v49H8z" fill={index === 1 ? '#6ebee0' : '#78c98b'} opacity=".72" />
        <path d="M25 74c3-18 12-27 23-26 8 1 10 11 18 14 6 3 10 1 14 3v25H25z" fill="#f4d35e" opacity=".88" />
        <path d="M43 71c2-11 7-16 13-13 5 2 5 8 12 11v17H43z" fill="#e45858" opacity=".9" />
      </g>
      <path d="M14 22l19 8 17-11 24 9" className="preview-boundary" />
      <text x="44" y="98" textAnchor="middle">{index === 2 ? 'CREF' : index === 1 ? 'Fenglei' : 'StormDiT'}</text>
    </g>)}
    {animated && <g className="preview-timeline">
      <path d="M36 153H284" />
      <circle cx="36" cy="153" r="4" className="preview-playhead" />
      <text x="36" y="168" textAnchor="middle">6 min</text>
      <text x="284" y="168" textAnchor="middle">180 min</text>
    </g>}
  </>
}

export function ChartPreview({ kind }: { kind: ChartPreviewKind }) {
  const [playing, setPlaying] = useState(false)
  let content
  if (kind === 'animation' || kind === 'spatial') {
    content = <SpatialPanels animated={kind === 'animation'} />
  } else if (kind === 'heatmap' || kind === 'correlation') {
    const colors = ['#315b83', '#7fa8c9', '#e7edf1', '#e5aa93', '#bd5f4d']
    content = <>
      <g transform="translate(66 20)">
        {Array.from({ length: 6 }).map((_, row) => Array.from({ length: 8 }).map((__, column) => (
          <rect key={`${row}-${column}`} x={column * 25} y={row * 20} width="23" height="18" rx="1.5" fill={colors[(row * 2 + column * 3) % colors.length]} />
        )))}
      </g>
      <path d="M66 151H266" className="preview-axis" /><path d="M53 20V140" className="preview-axis" />
      <text x="166" y="166" textAnchor="middle">{kind === 'correlation' ? '诊断变量' : '预报时效'}</text>
      <text x="18" y="82" textAnchor="middle" transform="rotate(-90 18 82)">{kind === 'correlation' ? '诊断变量' : '起报时间'}</text>
    </>
  } else if (kind === 'distribution') {
    content = <>
      <Axes />
      {[82, 142, 202, 262].map((x, index) => <g key={x}>
        <path d={`M${x} 37c-${12 + index * 2} 15-${13 + index} 31-4 45 8 12 9 26 4 44h8c-5-18-4-32 4-44 9-14 8-30-4-45Z`} fill={['#75aadb','#80c7ad','#e4af68','#cf7b7b'][index]} opacity=".72" />
        <path d={`M${x - 11} ${75 + index * 4}h22M${x} ${55 + index * 3}v55`} className="preview-data-line" />
      </g>)}
    </>
  } else if (kind === 'scatter') {
    content = <>
      <Axes />
      <path d="M38 142 284 24" className="preview-reference" />
      {[[67,126],[79,116],[96,119],[110,104],[126,105],[143,92],[158,88],[174,82],[192,73],[211,70],[230,58],[251,51],[268,43]].map(([x,y], index) => <circle key={x} cx={x} cy={y} r="4.5" fill={index < 5 ? '#4c91c5' : index < 9 ? '#56ae8a' : '#df8b58'} />)}
      <text x="165" y="166" textAnchor="middle">Observed rate</text>
    </>
  } else if (kind === 'summary') {
    content = <>
      <Axes />
      {[72,125,178,231].map((x, index) => <g key={x}>
        <rect x={x - 15} y={58 + index * 7} width="12" height={84 - index * 7} fill="#3f83ad" />
        <rect x={x + 2} y={42 + index * 11} width="12" height={100 - index * 11} fill="#d47a62" />
      </g>)}
      <path d="M57 48h12M57 54h12" stroke="#3f83ad" strokeWidth="3" /><text x="75" y="55">Model score</text>
    </>
  } else {
    content = <>
      <Axes />
      <path d="M44 45C82 48 104 57 132 70s54 17 78 31 48 17 76 27" className="preview-series preview-blue" />
      <path d="M44 61c34 2 60 9 89 19s50 13 77 20 47 14 76 19" className="preview-series preview-red" />
      {(kind === 'multi-line' || kind === 'fss') && <path d="M44 81c31-3 60 2 88 5s50 7 77 11 50 6 77 11" className="preview-series preview-green" />}
      {kind === 'fss' && <path d="M44 101c34-7 62-5 90-2s54 3 76 1 48 2 76 0" className="preview-series preview-gold" />}
      <text x="165" y="166" textAnchor="middle">预报时效（min）</text>
    </>
  }

  const chart = <svg className="chart-preview-svg" viewBox="0 0 320 184" role="img" aria-label="图表形式示意图">
      <title>图表形式示意图</title>
      {content}
    </svg>

  if (kind !== 'animation') return chart

  return <div className={`chart-preview-animation ${playing ? 'is-playing' : ''}`}>
    {chart}
    <button
      type="button"
      className="chart-preview-play"
      aria-label={playing ? '暂停同步动画示意' : '播放同步动画示意'}
      aria-pressed={playing}
      onClick={() => setPlaying((value) => !value)}
    >
      {playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}
    </button>
  </div>
}
