import ReactECharts from 'echarts-for-react'
import { useEffect, useMemo, useState } from 'react'
import type { CurveSeries, ThemeName } from '../types'

const modelColors: Record<string, string> = {
  stormdit: '#E15759', StormDiT: '#E15759', fenglei: '#ffad6f', '风雷': '#ffad6f',
  ConvLSTM: '#4ecdc4', PredRNN: '#59A14F', Earthformer: '#82aaff', NowCastNet: '#E6B800',
  SimVP: '#7bd88f', DiffCast: '#f78c6c', Persistence: '#8c96aa',
}
const modelLabels: Record<string, string> = {
  stormdit: 'StormDiT',
  fenglei: '风雷',
}

function modelColor(model: string) {
  return modelColors[model] || '#adc6ff'
}

export function MetricChart({ curves, theme }: { curves: CurveSeries[]; theme: ThemeName }) {
  const metrics = useMemo(() => [...new Set(curves.map((item) => item.metric))], [curves])
  const thresholds = useMemo(
    () => [...new Set(curves.map((item) => item.threshold_dbz))].sort((a, b) => a - b),
    [curves],
  )
  const [metric, setMetric] = useState(metrics[0] || 'csi')
  const [threshold, setThreshold] = useState(thresholds[0] || 20)
  const modelOrder = ['PredRNN', 'NowCastNet', 'StormDiT', 'stormdit', 'ConvLSTM', 'Earthformer', 'SimVP', 'DiffCast', 'Fenglei', 'fenglei', '风雷', 'Persistence']
  const selected = curves.filter((item) => item.metric === metric && item.threshold_dbz === threshold)
    .sort((a, b) => {
      const rank = (name: string) => modelOrder.includes(name) ? modelOrder.indexOf(name) : modelOrder.length
      return rank(a.model) - rank(b.model) || a.model.localeCompare(b.model)
    })
  const dark = theme === 'dark'

  useEffect(() => {
    if (!metrics.includes(metric)) setMetric(metrics[0] || 'csi')
  }, [metric, metrics])

  useEffect(() => {
    if (!thresholds.includes(threshold)) setThreshold(thresholds[0] || 20)
  }, [threshold, thresholds])
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    animationDuration: 720,
    animationEasing: 'cubicOut',
    textStyle: { fontFamily: 'Inter, Noto Sans SC, PingFang SC, sans-serif' },
    tooltip: {
      trigger: 'axis',
      backgroundColor: dark ? '#182330' : '#ffffff',
      borderColor: dark ? '#526171' : '#b9c5d0',
      padding: [10, 12],
      textStyle: { color: dark ? '#dce3ea' : '#1d2938', fontSize: 12 },
      axisPointer: { type: 'line', lineStyle: { color: dark ? '#71869a' : '#8fa4b5', type: 'dashed' } },
      extraCssText: `box-shadow: 0 4px 12px ${dark ? 'rgba(0, 0, 0, .25)' : 'rgba(22, 38, 56, .14)'}; border-radius: 4px;`,
    },
    legend: { show: false },
    grid: { left: 68, right: 28, top: 28, bottom: 60 },
    xAxis: {
      type: 'value',
      name: '预报时效（分钟）',
      nameLocation: 'middle',
      nameGap: 38,
      nameTextStyle: { color: dark ? '#9aa8b7' : '#607286', fontSize: 12 },
      axisLabel: { color: dark ? '#9aa8b7' : '#607286', fontSize: 11 },
      axisLine: { lineStyle: { color: dark ? '#4c5b6a' : '#aebbc7' } },
      axisTick: { lineStyle: { color: dark ? '#4c5b6a' : '#aebbc7' } },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: metric.toUpperCase(),
      nameLocation: 'middle',
      nameGap: 48,
      nameRotate: 90,
      min: ['csi', 'pod', 'far'].includes(metric) ? 0 : undefined,
      max: ['csi', 'pod', 'far'].includes(metric) ? 1 : undefined,
      nameTextStyle: { color: dark ? '#9aa8b7' : '#607286', fontSize: 12 },
      axisLabel: { color: dark ? '#9aa8b7' : '#607286', fontSize: 11 },
      axisLine: { show: true, lineStyle: { color: dark ? '#4c5b6a' : '#aebbc7' } },
      axisTick: { lineStyle: { color: dark ? '#4c5b6a' : '#aebbc7' } },
      splitLine: { show: true, lineStyle: { color: dark ? '#2d3a47' : '#e2e7ec', type: 'dashed' } },
    },
    series: selected.map((item) => ({
      name: item.model,
      type: 'line',
      data: item.points,
      connectNulls: false,
      smooth: 0.18,
      symbol: 'circle',
      symbolSize: 6,
      showSymbol: false,
      lineStyle: { width: 2.6, color: modelColor(item.model), shadowColor: modelColor(item.model), shadowBlur: 4 },
      itemStyle: { color: modelColor(item.model), borderColor: dark ? '#182330' : '#ffffff', borderWidth: 2 },
      emphasis: { focus: 'series', scale: 1.4, lineStyle: { width: 3.4 } },
    })),
  }), [selected, metric, dark])

  return (
    <div className="metric-chart">
      <div className="chart-toolbar">
        <div><strong>指标随预报时效变化</strong><span>{threshold} dBZ 阈值</span></div>
        <div>
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>{metrics.map((item) => <option key={item} value={item}>{item.toUpperCase()}</option>)}</select>
          <select value={threshold} onChange={(event) => setThreshold(Number(event.target.value))}>{thresholds.map((item) => <option key={item} value={item}>{item} dBZ</option>)}</select>
        </div>
      </div>
      <div className="chart-model-legend" aria-label="模型颜色图例">
        {selected.map((item) => (
          <span key={item.model}>
            <i style={{ backgroundColor: modelColor(item.model) }} />
            {modelLabels[item.model] || item.model}
          </span>
        ))}
      </div>
      <ReactECharts option={option} style={{ height: 370 }} notMerge />
    </div>
  )
}
