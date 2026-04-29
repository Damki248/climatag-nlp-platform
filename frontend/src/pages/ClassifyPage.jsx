import { useState } from 'react'
import axios from 'axios'

const EXAMPLE = "Melting Arctic permafrost is releasing significant amounts of methane, a potent greenhouse gas, accelerating global warming feedback loops."

export default function ClassifyPage() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleClassify() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.post('/api/classify/predict', { text, top_k: 5 })
      setResult(data)
    } catch (e) {
      setError('Classification request failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const maxScore = result?.top_k?.[0]?.score ?? 1

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Text Classification</h1>
        <p className="text-sm text-gray-500 mt-1">
          Classify climate research text into 20 topic categories (SciDCC, SciClimateBERT)
        </p>
      </div>

      <div className="space-y-3">
        <textarea
          className="w-full h-36 rounded-lg border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
          placeholder="Enter climate research text..."
          value={text}
          onChange={e => setText(e.target.value)}
        />
        <div className="flex gap-3">
          <button
            onClick={handleClassify}
            disabled={loading || !text.trim()}
            className="px-5 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 transition-colors"
          >
            {loading ? 'Classifying...' : 'Classify'}
          </button>
          <button
            onClick={() => setText(EXAMPLE)}
            className="px-5 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
          >
            Load example
          </button>
          {result && (
            <button
              onClick={() => setResult(null)}
              className="px-5 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {result && (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded-lg px-5 py-4">
            <p className="text-xs text-green-700 font-medium uppercase tracking-wide mb-1">Top prediction</p>
            <p className="text-xl font-bold text-green-900">{result.label}</p>
            <p className="text-sm text-green-700 mt-0.5">{(result.score * 100).toFixed(1)}% confidence</p>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 px-5 py-4 space-y-3">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Top 5 categories</p>
            {result.top_k.map((item, i) => (
              <div key={i} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className={i === 0 ? 'font-medium text-gray-900' : 'text-gray-600'}>{item.label}</span>
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
        </div>
      )}
    </div>
  )
}