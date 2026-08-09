import { LinesChart } from 'echarts/charts'
import { GeoComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import { useMemo } from 'react'
import chinaGeoJsonSource from '../../../visualization/assets/china_admin1_simplified.geojson?raw'
import type { ThemeName } from '../types'

const MAP_NAME = 'china-verification-region'
const chinaGeoJson = JSON.parse(chinaGeoJsonSource) as Parameters<typeof echarts.registerMap>[1]

echarts.use([LinesChart, GeoComponent, CanvasRenderer])
echarts.registerMap(MAP_NAME, chinaGeoJson)

interface ChinaLocatorMapProps {
  bbox: number[]
  theme: ThemeName
  isDockTarget?: boolean
  showRegion?: boolean
}

export function ChinaLocatorMap({
  bbox,
  theme,
  isDockTarget = false,
  showRegion = false,
}: ChinaLocatorMapProps) {
  const [south, north, west, east] = bbox
  const dark = theme === 'dark'

  const option = useMemo(() => ({
    animation: true,
    animationDuration: 450,
    animationDurationUpdate: 420,
    animationEasingUpdate: 'cubicOut',
    tooltip: { show: false },
    geo: {
      map: MAP_NAME,
      roam: false,
      silent: true,
      left: 8,
      right: 8,
      top: 8,
      bottom: 8,
      itemStyle: {
        areaColor: dark ? '#223344' : '#dce7ed',
        borderColor: dark ? '#567087' : '#8ea7b7',
        borderWidth: 0.65,
      },
      emphasis: { disabled: true },
    },
    series: [
      {
        type: 'lines',
        coordinateSystem: 'geo',
        polyline: true,
        silent: true,
        zlevel: 2,
        data: showRegion ? [{
          coords: [
            [west, south],
            [west, north],
            [east, north],
            [east, south],
            [west, south],
          ],
        }] : [],
        animationDuration: 680,
        animationDelay: showRegion ? 180 : 0,
        animationEasing: 'cubicOut',
        lineStyle: {
          color: dark ? '#ffd06b' : '#c95e26',
          width: 2.4,
          opacity: 1,
          shadowBlur: 7,
          shadowColor: dark ? 'rgba(255, 208, 107, .4)' : 'rgba(201, 94, 38, .25)',
        },
      },
    ],
  }), [dark, east, north, showRegion, south, west])

  return (
    <section
      className="locator-map"
      aria-label="当前检验区域"
      data-locator-map={isDockTarget ? true : undefined}
    >
      <div className="locator-map-title">
        <strong>检验区域</strong>
        <span>中国</span>
      </div>
      <ReactEChartsCore
        echarts={echarts}
        className="locator-map-chart"
        option={option}
        notMerge
        style={{ height: 210, width: '100%' }}
      />
      <div className="locator-coordinates">
        <span>{south.toFixed(1)}—{north.toFixed(1)}°N</span>
        <span>{west.toFixed(1)}—{east.toFixed(1)}°E</span>
      </div>
    </section>
  )
}
