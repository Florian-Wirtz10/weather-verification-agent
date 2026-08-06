import ReactECharts from 'echarts-for-react'
import { useMemo, useState } from 'react'
import type { CurveSeries } from '../types'

const modelColors: Record<string, string> = {
  stormdit: '#65a4ff', StormDiT: '#65a4ff', fenglei: '#ffad6f', '风雷': '#ffad6f',
  ConvLSTM: '#4ecdc4', PredRNN: '#c792ea', Earthformer: '#82aaff', NowCastNet: '#f6c177',
  SimVP: '#7bd88f', DiffCast: '#f78c6c', Persistence: '#8c96aa',
}

export function MetricChart({ curves }: { curves: CurveSeries[] }) {
  const metrics = [...new Set(curves.map((item) => item.metric))]
  const thresholds = [...new Set(curves.map((item) => item.threshold_dbz))].sort((a, b) => a - b)
  const [metric, setMetric] = useState(metrics[0] || 'csi')
  const [threshold, setThreshold] = useState(thresholds[0] || 20)
  const selected = curves.filter((item) => item.metric === metric && item.threshold_dbz === threshold)
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    animationDuration: 500,
    tooltip: { trigger: 'axis', backgroundColor: '#171f33', borderColor: '#424754', textStyle: { color: '#dae2fd' } },
    legend: { top: 0, right: 4, textStyle: { color: '#c2c6d6' } },
    grid: { left: 48, right: 22, top: 42, bottom: 42 },
    xAxis: { type: 'value', name: 'Lead time (min)', nameTextStyle: { color: '#8c909f' }, axisLabel: { color: '#8c909f' }, axisLine: { lineStyle: { color: '#424754' } }, splitLine: { show: false } },
    yAxis: { type: 'value', name: metric.toUpperCase(), min: ['csi', 'pod', 'far'].includes(metric) ? 0 : undefined, max: ['csi', 'pod', 'far'].includes(metric) ? 1 : undefined, nameTextStyle: { color: '#8c909f' }, axisLabel: { color: '#8c909f' }, axisLine: { show: true, lineStyle: { color: '#424754' } }, splitLine: { show: false } },
    series: selected.map((item) => ({
      name: item.model,
      type: 'line',
      data: item.points,
      connectNulls: false,
      smooth: 0.18,
      symbol: 'circle',
      symbolSize: 5,
      lineStyle: { width: 2.2, color: modelColors[item.model] || '#adc6ff' },
      itemStyle: { color: modelColors[item.model] || '#adc6ff' },
    })),
  }), [selected, metric])

  return (
    <div className="metric-chart">
      <div className="chart-toolbar">
        <div><strong>指标随预报时效变化</strong><span>交互曲线 · 无底部网格线</span></div>
        <div>
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>{metrics.map((item) => <option key={item}>{item.toUpperCase()}</option>)}</select>
          <select value={threshold} onChange={(event) => setThreshold(Number(event.target.value))}>{thresholds.map((item) => <option key={item} value={item}>{item} dBZ</option>)}</select>
        </div>
      </div>
      <ReactECharts option={option} style={{ height: 330 }} notMerge />
    </div>
  )
}
