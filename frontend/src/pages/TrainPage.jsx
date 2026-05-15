import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const POLL_INTERVAL = 2000

function StatusBadge({ status }) {
  const map = {
    idle:     'bg-gray-100 text-gray-600',
    running:  'bg-blue-100 text-blue-700',
    finished: 'bg-green-100 text-green-700',
    failed:   'bg-red-100 text-red-700',
  }
  return (
    <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${map[status] ?? 'bg-gray-100 text-gray-500'}`}>
      {status}
    </span>
  )
}

export default function TrainPage() {
  const [config, setConfig] = useState({
    epochs:       10,
    lr:           5e-6,
    batch:        8,
    silver_ratio: 5,
    run_name:     '',
  })
  const [status, setStatus]         = useState({ status: 'idle', logs: [] })
  const [annotationCount, setAnnotationCount] = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const logsRef = useRef(null)
  const pollRef = useRef(null)

  useEffect(() => {
    fetchStatus()
    fetchAnnotationCount()
    return () => clearInterval(pollRef.current)
  }, [])

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [status.logs])

  async function fetchStatus() {
    try {
      const { data } = await axios.get('/api/train/status')
      setStatus(data)
      if (data.status === 'running') startPolling()
    } catch {}
  }

  async function fetchAnnotationCount() {
    try {
      const { data } = await axios.get('/api/annotation/export')
      setAnnotationCount(data.task_count ?? 0)
    } catch {
      setAnnotationCount(null)
    }
  }

  function startPolling() {
    clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await axios.get('/api/train/status')
        setStatus(data)
        if (data.status !== 'running') clearInterval(pollRef.current)
      } catch {}
    }, POLL_INTERVAL)
  }

  async function handleStart() {
    setLoading(true)
    setError(null)
    try {
      await axios.post('/api/train/start', {
        ...config,
        run_name: config.run_name || undefined,
      })
      startPolling()
      await fetchStatus()
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleReset() {
    try {
      await axios.post('/api/train/reset')
      setStatus({ status: 'idle', logs: [] })
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message)
    }
  }

  const isRunning  = status.status === 'running'
  const isFinished = status.status === 'finished'
  const isFailed   = status.status === 'failed'
  const progress   = status.total_epochs > 0
    ? Math.round((status.current_epoch / status.total_epochs) * 100)
    : 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Fine-tune GLiNER</h1>
        <p className="text-sm text-gray-500 mt-1">
          Train the GLiNER model on Climate Model annotations from Label Studio
        </p>
      </div>

      {/* Dataset info */}
      <div className="bg-white rounded-lg border border-gray-200 px-5 py-4 flex items-center justify-between">
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Training dataset</p>
          <p className="text-sm text-gray-700">
            Climate Model annotations from Label Studio
          </p>
          <p className="text-xs text-gray-400 mt-0.5">
            Combined with CliReNER SILVER dataset using experience replay
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Annotated samples</p>
          <p className="text-2xl font-bold text-green-700">
            {annotationCount !== null ? annotationCount : '—'}
          </p>
          <button
            onClick={fetchAnnotationCount}
            className="text-xs text-gray-400 hover:text-gray-600 mt-0.5"
          >
            refresh
          </button>
        </div>
      </div>

      {/* Config form */}
      <div className="bg-white rounded-lg border border-gray-200 p-5 space-y-4">
        <p className="text-sm font-medium text-gray-700">Training configuration</p>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Epochs</label>
            <input
              type="number" min={1} max={50}
              value={config.epochs}
              onChange={e => setConfig(c => ({ ...c, epochs: +e.target.value }))}
              disabled={isRunning}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Learning rate</label>
            <input
              type="number" step="1e-7" min={1e-7} max={1e-4}
              value={config.lr}
              onChange={e => setConfig(c => ({ ...c, lr: +e.target.value }))}
              disabled={isRunning}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Batch size</label>
            <select
              value={config.batch}
              onChange={e => setConfig(c => ({ ...c, batch: +e.target.value }))}
              disabled={isRunning}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            >
              {[4, 8, 16].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">
              Silver ratio
              <span className="text-gray-400 ml-1">(SILVER : Climate Model)</span>
            </label>
            <select
              value={config.silver_ratio}
              onChange={e => setConfig(c => ({ ...c, silver_ratio: +e.target.value }))}
              disabled={isRunning}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            >
              {[3, 5, 8, 10].map(v => <option key={v} value={v}>{v}:1</option>)}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">
            Run name <span className="text-gray-400">(optional)</span>
          </label>
          <input
            type="text"
            placeholder="e.g. gliner_v2_lr5e6"
            value={config.run_name}
            onChange={e => setConfig(c => ({ ...c, run_name: e.target.value }))}
            disabled={isRunning}
            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
          />
        </div>

        <div className="flex gap-3 pt-1">
          <button
            onClick={handleStart}
            disabled={isRunning || loading || annotationCount === 0}
            className="px-5 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 transition-colors"
            title={annotationCount === 0 ? 'No annotations in Label Studio' : ''}
          >
            {loading ? 'Starting...' : isRunning ? 'Training...' : 'Start training'}
          </button>
          {(isFinished || isFailed) && (
            <button
              onClick={handleReset}
              className="px-5 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
            >
              Reset
            </button>
          )}
        </div>

        {annotationCount === 0 && (
          <p className="text-xs text-amber-600">
            No annotations found in Label Studio. Add annotations via the Annotate tab first.
          </p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {/* Status panel */}
      {status.status !== 'idle' && (
        <div className="bg-white rounded-lg border border-gray-200 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <StatusBadge status={status.status} />
              {status.run_name && (
                <span className="text-sm text-gray-600 font-mono">{status.run_name}</span>
              )}
            </div>
            {isRunning && (
              <span className="text-sm text-gray-500">
                Epoch {status.current_epoch} / {status.total_epochs}
              </span>
            )}
          </div>

          {(isRunning || isFinished) && (
            <div>
              <div className="w-full bg-gray-100 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${isFinished ? 'bg-green-600' : 'bg-blue-500'}`}
                  style={{ width: `${isFinished ? 100 : progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-1">{isFinished ? 100 : progress}% complete</p>
            </div>
          )}

          {status.logs?.length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Logs</p>
              <div
                ref={logsRef}
                className="bg-gray-900 rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs text-gray-300 space-y-0.5"
              >
                {status.logs.map((line, i) => (
                  <div key={i} className={
                    line.includes('✅') ? 'text-green-400' :
                    line.includes('ERROR') || line.includes('Error') ? 'text-red-400' :
                    line.includes('INFO') ? 'text-blue-300' :
                    'text-gray-300'
                  }>
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}

          {isFailed && status.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
              {status.error}
            </div>
          )}

          {isFinished && (
            <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-3 text-sm text-green-700">
              Training complete! New model saved to <code className="font-mono">models/ner_gliner_climate_model</code>.
              Restart the backend to load the updated model.
            </div>
          )}
        </div>
      )}
    </div>
  )
}