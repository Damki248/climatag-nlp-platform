import { useState, useEffect } from 'react'
import axios from 'axios'
import NEREditor from '../components/ner/NEREditor'

const EXAMPLE = "Arctic sea ice extent has declined significantly due to rising temperatures, with CO2 concentrations at the Mauna Loa Observatory reaching record highs."

export default function NERPage() {
  const [text, setText] = useState('')
  const [entities, setEntities] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [status, setStatus] = useState(null)
  const [switching, setSwitching] = useState(false)

  useEffect(() => {
    fetchStatus()
  }, [])

  async function fetchStatus() {
    try {
      const { data } = await axios.get('/api/ner/status')
      setStatus(data)
    } catch {}
  }

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

  async function handleSwitch(modelName) {
    setSwitching(true)
    try {
      await axios.post('/api/ner/switch', { model: modelName })
      await fetchStatus()
      setEntities([])
    } catch (e) {
      setError(`Switch failed: ${e.response?.data?.detail ?? e.message}`)
    } finally {
      setSwitching(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Named Entity Recognition</h1>
          <p className="text-sm text-gray-500 mt-1">
            Detect climate-domain entities using GLiNER (28 categories + Climate Model)
          </p>
        </div>

        {/* Model switch */}
        {status && (
          <div className="flex items-center gap-2 bg-white border border-gray-200 rounded-lg px-4 py-2">
            <span className="text-xs text-gray-500 mr-1">Model:</span>
            <button
              onClick={() => handleSwitch('baseline')}
              disabled={switching || status.active_model === 'baseline'}
              className={`text-xs px-3 py-1 rounded-md font-medium transition-colors ${
                status.active_model === 'baseline'
                  ? 'bg-green-700 text-white'
                  : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              {switching && status.active_model !== 'baseline' ? 'Switching...' : 'Baseline'}
            </button>
            <button
              onClick={() => handleSwitch('climate_model')}
              disabled={switching || status.active_model === 'climate_model' || !status.climate_model_available}
              className={`text-xs px-3 py-1 rounded-md font-medium transition-colors ${
                status.active_model === 'climate_model'
                  ? 'bg-green-700 text-white'
                  : !status.climate_model_available
                  ? 'text-gray-300 cursor-not-allowed'
                  : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
              }`}
              title={!status.climate_model_available ? 'Climate Model not available' : ''}
            >
              {switching && status.active_model !== 'climate_model' ? 'Switching...' : 'Climate Model'}
            </button>
          </div>
        )}
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