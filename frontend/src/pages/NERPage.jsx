import { useState } from 'react'
import axios from 'axios'
import NEREditor from '../components/ner/NEREditor'

const EXAMPLE = "Arctic sea ice extent has declined significantly due to rising temperatures, with CO2 concentrations at the Mauna Loa Observatory reaching record highs."

export default function NERPage() {
  const [text, setText] = useState('')
  const [entities, setEntities] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleAnalyze() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.post('/api/ner/predict', { text })
      setEntities(data.entities)
    } catch (e) {
      setError('NER request failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Named Entity Recognition</h1>
        <p className="text-sm text-gray-500 mt-1">
          Detect climate-domain entities using CliReNER (28 categories, SpanMarker + CliSciBERT)
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
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="px-5 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 transition-colors"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
          <button
            onClick={() => setText(EXAMPLE)}
            className="px-5 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
          >
            Load example
          </button>
          {entities.length > 0 && (
            <button
              onClick={() => setEntities([])}
              className="px-5 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {entities.length > 0 && (
        <NEREditor text={text} entities={entities} />
      )}
    </div>
  )
}