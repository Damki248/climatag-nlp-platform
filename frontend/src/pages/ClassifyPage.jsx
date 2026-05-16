import { useState } from 'react'

const LABELS = [
  'Agriculture & Food', 'Animals', 'Biology', 'Biotechnology', 'Climate',
  'Earthquakes', 'Endangered Animals', 'Environment', 'Extinction',
  'Genetically Modified', 'Geography', 'Geology', 'Global Warming',
  'Hurricanes Cyclones', 'Microbes', 'New Species', 'Ozone Holes',
  'Pollution', 'Weather', 'Zoology',
]

const MODEL_INFO = {
  name: 'SciClimateBERT',
  source: 'P0L3/SciClimateBERT',
  strategy: 'Full Fine-Tuning',
  dataset: 'SciDCC (20 categories, 11,400 articles)',
  testF1: '0.4208',
  run: 'full_ft_body_lr2e5_batch16_exp07',
}

export default function ClassifyPage() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleClassify() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch('/api/cls/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, top_k: 5 }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Classification failed')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleClassify()
  }

  const maxScore = result ? result.top_k[0].score : 1

  return (
    <div className="space-y-8">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Text Classification</h1>
        <p className="text-sm text-gray-500 mt-1">
          Classify scientific climate texts into 20 topic categories using SciClimateBERT.
        </p>
      </div>

      {/* Model info card */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <p className="text-xs font-semibold text-green-700 uppercase tracking-wide mb-2">Active Model</p>
        <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm">
          <div><span className="text-gray-500">Model:</span> <span className="font-medium text-gray-800">{MODEL_INFO.name}</span></div>
          <div><span className="text-gray-500">Strategy:</span> <span className="font-medium text-gray-800">{MODEL_INFO.strategy}</span></div>
          <div><span className="text-gray-500">Dataset:</span> <span className="font-medium text-gray-800">{MODEL_INFO.dataset}</span></div>
          <div><span className="text-gray-500">Test macro F1:</span> <span className="font-medium text-gray-800">{MODEL_INFO.testF1}</span></div>
        </div>
      </div>

      {/* Input */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-gray-700">
          Input Text
        </label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={6}
          placeholder="Paste a scientific abstract or article body here... (Ctrl+Enter to classify)"
          className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none"
        />
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">{text.length} characters</span>
          <button
            onClick={handleClassify}
            disabled={loading || !text.trim()}
            className="px-5 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Classifying...' : 'Classify'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">

          {/* Top prediction */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Top Prediction</p>
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-bold text-green-700">{result.prediction}</span>
              <span className="text-sm text-gray-500">
                {(result.score * 100).toFixed(1)}% confidence
              </span>
            </div>
          </div>

          {/* Top 5 breakdown */}
          <div className="bg-white border border-gray-200 rounded-lg p-5 space-y-3">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Top 5 Categories</p>
            {result.top_k.map((item, i) => (
              <div key={item.label} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className={`font-medium ${i === 0 ? 'text-green-700' : 'text-gray-700'}`}>
                    {item.label}
                  </span>
                  <span className="text-gray-500">{(item.score * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all ${i === 0 ? 'bg-green-600' : 'bg-gray-400'}`}
                    style={{ width: `${(item.score / maxScore) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* All categories reference */}
          <details className="bg-white border border-gray-200 rounded-lg">
            <summary className="px-5 py-3 text-xs font-semibold text-gray-400 uppercase tracking-wide cursor-pointer select-none hover:text-gray-600">
              All 20 Categories
            </summary>
            <div className="px-5 pb-4 flex flex-wrap gap-2">
              {LABELS.map(label => (
                <span
                  key={label}
                  className={`text-xs px-2 py-1 rounded-full border ${
                    label === result.prediction
                      ? 'bg-green-100 border-green-300 text-green-700 font-medium'
                      : 'bg-gray-50 border-gray-200 text-gray-500'
                  }`}
                >
                  {label}
                </span>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}