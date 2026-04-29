import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const MLFLOW_URL = 'http://localhost:5000'

// MLflow vraća metrike i parametre kao array [{key, value}]
// Konvertiramo u dictionary {key: value} za lakše korištenje
function normalizeRun(run) {
  const metrics = {}
  for (const { key, value } of run.data?.metrics ?? []) {
    metrics[key] = value
  }
  const params = {}
  for (const { key, value } of run.data?.params ?? []) {
    params[key] = value
  }
  return { ...run, data: { metrics, params } }
}

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 px-5 py-4">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-xl font-bold text-gray-900">{value ?? '—'}</p>
      {sub && <p className="text-sm text-gray-500 mt-0.5 truncate" title={sub}>{sub}</p>}
    </div>
  )
}

function StatusBadge({ status }) {
  const map = {
    FINISHED: 'bg-green-100 text-green-700',
    RUNNING:  'bg-blue-100 text-blue-700',
    FAILED:   'bg-red-100 text-red-700',
    KILLED:   'bg-gray-100 text-gray-500',
  }
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${map[status] ?? 'bg-gray-100 text-gray-500'}`}>
      {status}
    </span>
  )
}

function MetricBar({ value, max }) {
  if (value == null) return <span className="text-gray-400 text-sm">—</span>
  const pct = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 bg-gray-100 rounded-full h-1.5">
        <div className="h-1.5 rounded-full bg-green-600" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-sm text-gray-700 font-mono">{(value * 100).toFixed(1)}%</span>
    </div>
  )
}

function RunRow({ run, bestF1, isProduction, onClick, isSelected }) {
  const p = run.data?.params ?? {}
  const m = run.data?.metrics ?? {}
  const name = run.info?.run_name ?? run.info?.run_id?.slice(0, 8)
  const status = run.info?.status

  return (
    <tr
      onClick={() => onClick(run)}
      className={`cursor-pointer transition-colors border-b border-gray-100 last:border-0
        ${isSelected ? 'bg-green-50' : 'hover:bg-gray-50'}`}
    >
      <td className="px-4 py-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-medium text-gray-900">{name}</span>
          {isProduction && (
            <span className="text-xs bg-green-700 text-white px-2 py-0.5 rounded-full">production</span>
          )}
        </div>
        <p className="text-xs text-gray-400 mt-0.5">{p.strategy ?? '—'} · {p.input ?? '—'}</p>
      </td>
      <td className="px-4 py-3">
        <StatusBadge status={status} />
      </td>
      <td className="px-4 py-3">
        <MetricBar value={m.test_macro_f1} max={bestF1} />
      </td>
      <td className="px-4 py-3 text-sm text-gray-600 font-mono">
        {m.val_macro_f1 != null ? (m.val_macro_f1 * 100).toFixed(1) + '%' : '—'}
      </td>
      <td className="px-4 py-3 text-sm text-gray-600">
        {p.lr ?? '—'}
      </td>
      <td className="px-4 py-3 text-sm text-gray-600">
        {p.epochs ?? '—'}
      </td>
    </tr>
  )
}

function RunDetail({ run, onClose }) {
  const p = run.data?.params ?? {}
  const m = run.data?.metrics ?? {}
  const name = run.info?.run_name ?? run.info?.run_id?.slice(0, 8)

  // Extract per-class F1 metrics
  const perClass = Object.entries(m)
    .filter(([k]) => k.startsWith('test_f1_'))
    .map(([k, v]) => ({ label: k.replace('test_f1_', '').replace(/_/g, ' '), value: v }))
    .sort((a, b) => b.value - a.value)

  const maxPerClass = perClass[0]?.value ?? 1

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-gray-900">{name}</h3>
          <p className="text-xs text-gray-400 font-mono mt-0.5">{run.info?.run_id}</p>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 text-xl leading-none"
        >
          ×
        </button>
      </div>

      {/* Test metrics */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Test macro F1', value: m.test_macro_f1 != null ? (m.test_macro_f1 * 100).toFixed(2) + '%' : '—' },
          { label: 'Test weighted F1', value: m.test_weighted_f1 != null ? (m.test_weighted_f1 * 100).toFixed(2) + '%' : '—' },
          { label: 'Test accuracy', value: m.test_accuracy != null ? (m.test_accuracy * 100).toFixed(2) + '%' : '—' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-50 rounded-lg px-4 py-3">
            <p className="text-xs text-gray-500 mb-1">{label}</p>
            <p className="text-base font-bold text-gray-900">{value}</p>
          </div>
        ))}
      </div>

      {/* Params */}
      <div>
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Parameters</p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          {Object.entries(p).map(([k, v]) => (
            <div key={k} className="flex justify-between text-sm py-0.5 border-b border-gray-100">
              <span className="text-gray-500">{k}</span>
              <span className="text-gray-900 font-mono text-xs">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Per-class F1 */}
      {perClass.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Per-class F1</p>
          <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
            {perClass.map(({ label, value }) => (
              <div key={label} className="flex items-center gap-3">
                <span className="text-xs text-gray-600 w-32 truncate capitalize">{label}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                  <div
                    className="h-1.5 rounded-full bg-green-500"
                    style={{ width: `${(value / maxPerClass) * 100}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-gray-700 w-10 text-right">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function ExperimentsPage() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedRun, setSelectedRun] = useState(null)
  const [productionRunId, setProductionRunId] = useState(null)
  const [sortBy, setSortBy] = useState('test_macro_f1')

  const fetchRuns = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // fetch experiments
      const expRes = await axios.get(`${MLFLOW_URL}/api/2.0/mlflow/experiments/search`, {
        params: { max_results: 10 }
      })
      const experiments = expRes.data.experiments ?? []

      if (experiments.length === 0) {
        setRuns([])
        return
      }

      // fetch runs from all experiments
      const allRuns = []
      for (const exp of experiments) {
        const runsRes = await axios.post(`${MLFLOW_URL}/api/2.0/mlflow/runs/search`, {
          experiment_ids: [exp.experiment_id],
          max_results: 50,
          order_by: ['attribute.start_time DESC'],
        })
        allRuns.push(...(runsRes.data.runs ?? []).map(normalizeRun))
      }

      setRuns(allRuns)

      // try to find production model run
      try {
        const mvRes = await axios.get(
          `${MLFLOW_URL}/api/2.0/mlflow/registered-models/alias`,
          { params: { name: 'SciDCC-Classifier', alias: 'production' } }
        )
        setProductionRunId(mvRes.data.model_version?.run_id ?? null)
      } catch {
        setProductionRunId(null)
      }
    } catch (e) {
      setError('Cannot connect to MLflow at ' + MLFLOW_URL + '. Is it running?')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchRuns() }, [fetchRuns])

  const sortedRuns = [...runs].sort((a, b) => {
    const ma = a.data?.metrics ?? {}
    const mb = b.data?.metrics ?? {}
    if (sortBy === 'test_macro_f1') return (mb.test_macro_f1 ?? -1) - (ma.test_macro_f1 ?? -1)
    if (sortBy === 'start_time') return (b.info?.start_time ?? 0) - (a.info?.start_time ?? 0)
    return 0
  })

  const bestF1 = Math.max(...runs.map(r => r.data?.metrics?.test_macro_f1 ?? 0), 0)
  const finishedRuns = runs.filter(r => r.info?.status === 'FINISHED')
  const runningRuns = runs.filter(r => r.info?.status === 'RUNNING')
  const bestRun = finishedRuns.reduce((best, r) => {
    const f1 = r.data?.metrics?.test_macro_f1 ?? 0
    return f1 > (best?.data?.metrics?.test_macro_f1 ?? -1) ? r : best
  }, null)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
          <p className="text-sm text-gray-500 mt-1">
            SciDCC classification runs tracked in MLflow
          </p>
        </div>
        <div className="flex gap-2">
          <a
            href={MLFLOW_URL}
            target="_blank"
            rel="noreferrer"
            className="px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
          >
            Open MLflow ↗
          </a>
          <button
            onClick={fetchRuns}
            disabled={loading}
            className="px-4 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 transition-colors"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Total runs" value={runs.length} />
        <StatCard label="Finished" value={finishedRuns.length} />
        <StatCard
          label="Running"
          value={runningRuns.length}
          sub={runningRuns.length > 0 ? 'in progress' : null}
        />
        <StatCard
          label="Best macro F1"
          value={bestF1 > 0 ? (bestF1 * 100).toFixed(2) + '%' : '—'}
          sub={bestRun?.info?.run_name ?? null}
        />
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-5 py-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {runs.length === 0 && !loading && !error && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg px-5 py-10 text-center text-gray-500 text-sm">
          No runs found. Start a training run to see results here.
        </div>
      )}

      {runs.length > 0 && (
        <div className="space-y-4">
          {/* Sort control */}
          <div className="flex items-center gap-3 text-sm">
            <span className="text-gray-500">Sort by:</span>
            {[
              { key: 'test_macro_f1', label: 'Best F1' },
              { key: 'start_time', label: 'Most recent' },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSortBy(key)}
                className={`px-3 py-1 rounded-md transition-colors ${
                  sortBy === key
                    ? 'bg-green-100 text-green-700 font-medium'
                    : 'text-gray-500 hover:text-gray-900'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Runs table */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  {['Run', 'Status', 'Test macro F1', 'Val macro F1', 'LR', 'Epochs'].map(h => (
                    <th key={h} className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedRuns.map(run => (
                  <RunRow
                    key={run.info?.run_id}
                    run={run}
                    bestF1={bestF1}
                    isProduction={run.info?.run_id === productionRunId}
                    onClick={setSelectedRun}
                    isSelected={selectedRun?.info?.run_id === run.info?.run_id}
                  />
                ))}
              </tbody>
            </table>
          </div>

          {/* Run detail panel */}
          {selectedRun && (
            <RunDetail
              run={selectedRun}
              onClose={() => setSelectedRun(null)}
            />
          )}
        </div>
      )}
    </div>
  )
}
