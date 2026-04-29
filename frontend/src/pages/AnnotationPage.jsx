import { useState } from 'react'
import axios from 'axios'
import NERAnnotator from '../components/annotation/NERAnnotator'

export default function AnnotationPage() {
  const [text, setText] = useState('')
  const [entities, setEntities] = useState(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handlePreannotate() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setEntities(null)
    try {
      const { data } = await axios.post('/api/ner/predict', { text })
      // dodaj id svakom entitetu za tracking
      setEntities(data.entities.map((e, i) => ({ ...e, id: i })))
    } catch {
      setError('Pre-annotation failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  async function handleSave(finalEntities) {
    setSaving(true)
    setError(null)
    try {
      // konvertiramo u Label Studio format kroz backend
      const { data } = await axios.post('/api/annotation/upload', {
        texts: [text],
        annotations: [finalEntities.map(e => ({
          span: e.span,
          label: e.label,
          start: e.start,
          end: e.end,
        }))]
      })
      setResult(data)
    } catch {
      setError('Save failed. Is the backend running?')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Annotation</h1>
        <p className="text-sm text-gray-500 mt-1">
          Pre-annotate with the NER model, correct manually, then save to Label Studio.
        </p>
      </div>

      {!entities && (
        <div className="space-y-3">
          <textarea
            className="w-full h-36 rounded-lg border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
            placeholder="Enter climate research text..."
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <div className="flex gap-3">
            <button
              onClick={handlePreannotate}
              disabled={loading || !text.trim()}
              className="px-5 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 transition-colors"
            >
              {loading ? 'Annotating...' : 'Pre-annotate'}
            </button>
            <button
              onClick={() => setText('Researchers at the University of Alaska measured permafrost temperatures across the Arctic tundra, finding that methane emissions increased by 30% over the past decade due to rising atmospheric CO2 levels.')}
              className="px-5 py-2 border border-gray-300 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
            >
              Load example
            </button>
          </div>
          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
      )}

      {entities && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-600 font-medium">
              {entities.length} entities pre-annotated — review and correct below
            </p>
            <button
              onClick={() => { setEntities(null); setResult(null) }}
              className="text-sm text-gray-400 hover:text-gray-600"
            >
              ← Back
            </button>
          </div>
          <NERAnnotator
            text={text}
            initialEntities={entities}
            onSave={handleSave}
          />
          {saving && <p className="text-sm text-gray-500">Saving...</p>}
          {error && <p className="text-sm text-red-600">{error}</p>}
          {result && (
            <div className="bg-green-50 border border-green-200 rounded-lg px-5 py-4 space-y-2">
              <p className="text-sm font-medium text-green-900">
                {result.uploaded} task saved to Label Studio
              </p>
              <a href={result.url} target="_blank" rel="noreferrer" className="text-sm text-green-700 underline hover:text-green-900">
                {'Open in Label Studio →'}
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  )
}