import { useState, useRef, useCallback } from 'react'

const LABELS = [
  'Asset', 'Body of Water', 'Body Part', 'Chemical', 'Date', 'Device',
  'Disease', 'Ecosystem', 'Event', 'Geographical Feature', 'Location',
  'Measuring Device', 'Meteorological Phenomenon', 'Money', 'Natural Disaster',
  'Natural Phenomenon', 'Numerical Value', 'Organism', 'Organization', 'Person',
  'Policy', 'Quantity', 'Research', 'Sensor', 'Species', 'Technology', 'Time', 'URL'
]

const LABEL_COLORS = {
  'Asset': 'bg-red-100 text-red-800 border-red-300',
  'Body of Water': 'bg-blue-100 text-blue-800 border-blue-300',
  'Body Part': 'bg-orange-100 text-orange-800 border-orange-300',
  'Chemical': 'bg-purple-100 text-purple-800 border-purple-300',
  'Date': 'bg-teal-100 text-teal-800 border-teal-300',
  'Device': 'bg-yellow-100 text-yellow-800 border-yellow-300',
  'Disease': 'bg-red-200 text-red-900 border-red-400',
  'Ecosystem': 'bg-green-100 text-green-800 border-green-300',
  'Event': 'bg-indigo-100 text-indigo-800 border-indigo-300',
  'Geographical Feature': 'bg-violet-100 text-violet-800 border-violet-300',
  'Location': 'bg-cyan-100 text-cyan-800 border-cyan-300',
  'Measuring Device': 'bg-amber-100 text-amber-800 border-amber-300',
  'Meteorological Phenomenon': 'bg-sky-100 text-sky-800 border-sky-300',
  'Money': 'bg-yellow-100 text-yellow-900 border-yellow-400',
  'Natural Disaster': 'bg-red-200 text-red-900 border-red-400',
  'Natural Phenomenon': 'bg-teal-100 text-teal-900 border-teal-400',
  'Numerical Value': 'bg-gray-100 text-gray-700 border-gray-300',
  'Organism': 'bg-green-100 text-green-900 border-green-400',
  'Organization': 'bg-blue-100 text-blue-900 border-blue-400',
  'Person': 'bg-violet-100 text-violet-900 border-violet-400',
  'Policy': 'bg-orange-100 text-orange-900 border-orange-400',
  'Quantity': 'bg-yellow-100 text-yellow-800 border-yellow-300',
  'Research': 'bg-cyan-100 text-cyan-900 border-cyan-400',
  'Sensor': 'bg-red-100 text-red-800 border-red-300',
  'Species': 'bg-blue-100 text-blue-800 border-blue-300',
  'Technology': 'bg-purple-100 text-purple-900 border-purple-400',
  'Time': 'bg-orange-100 text-orange-800 border-orange-300',
  'URL': 'bg-gray-100 text-gray-600 border-gray-300',
}

const DEFAULT_COLOR = 'bg-gray-100 text-gray-700 border-gray-300'

function buildSegments(text, entities) {
  const sorted = [...entities].sort((a, b) => a.start - b.start)
  const segments = []
  let cursor = 0
  for (const ent of sorted) {
    if (ent.start > cursor) {
      segments.push({ type: 'text', content: text.slice(cursor, ent.start) })
    }
    if (ent.start >= cursor) {
      segments.push({ type: 'entity', content: text.slice(ent.start, ent.end), entity: ent })
      cursor = ent.end
    }
  }
  if (cursor < text.length) {
    segments.push({ type: 'text', content: text.slice(cursor) })
  }
  return segments
}

export default function NERAnnotator({ text, initialEntities = [], onSave }) {
  const [entities, setEntities] = useState(initialEntities)
  const [menu, setMenu] = useState(null)
  const textRef = useRef(null)

  const handleMouseUp = useCallback((e) => {
    e.stopPropagation()

    const selection = window.getSelection()
    if (!selection || selection.isCollapsed) return

    const range = selection.getRangeAt(0)
    const container = textRef.current
    if (!container || !container.contains(range.commonAncestorContainer)) return

    const selectedText = selection.toString().trim()
    if (!selectedText) return

    // nalazimo start offset hodajuci kroz text nodeove kontejnera
    // preskacemo nodeove koji imaju data-label="true" (labelice entiteta)
    function getOffset(targetNode, targetOffset) {
        let offset = 0
        const walker = document.createTreeWalker(
        container,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode(node) {
            // preskoči text nodeove unutar label spanova
            const parent = node.parentElement
            if (parent && parent.dataset.label === 'true') {
                return NodeFilter.FILTER_REJECT
            }
            return NodeFilter.FILTER_ACCEPT
            }
        }
        )
        while (walker.nextNode()) {
        const node = walker.currentNode
        if (node === targetNode) {
            return offset + targetOffset
        }
        offset += node.textContent.length
        }
        return offset
    }

    const start = getOffset(range.startContainer, range.startOffset)
    const end = start + selectedText.length

    selection.removeAllRanges()

    const overlaps = entities.some(e => !(end <= e.start || start >= e.end))
    if (overlaps) return

    const rect = range.getBoundingClientRect()
    setMenu({
        type: 'new',
        x: rect.left,
        y: rect.bottom + 4,
        start,
        end,
        span: selectedText,
    })
    }, [entities])

  function handleEntityClick(e, entity) {
    e.stopPropagation()
    const rect = e.currentTarget.getBoundingClientRect()
    setMenu({
      type: 'edit',
      x: rect.left,
      y: rect.bottom + 4,
      entityId: entity.id,
      span: entity.span,
      currentLabel: entity.label,
    })
  }

  function addEntity(label) {
    if (!menu || menu.type !== 'new') return
    setEntities(prev => [...prev, {
      id: Date.now(),
      span: menu.span,
      label,
      start: menu.start,
      end: menu.end,
      score: 1.0,
    }])
    setMenu(null)
  }

  function changeLabel(label) {
    setEntities(prev => prev.map(e => e.id === menu.entityId ? { ...e, label } : e))
    setMenu(null)
  }

  function deleteEntity() {
    setEntities(prev => prev.filter(e => e.id !== menu.entityId))
    setMenu(null)
  }

  const segments = buildSegments(text, entities)

  return (
    <>
      <div className="space-y-4">
        <div
          ref={textRef}
          onMouseUp={handleMouseUp}
          className="bg-white rounded-lg border border-gray-200 px-5 py-4 text-sm leading-8 select-text cursor-text"
        >
          {segments.map((seg, i) => {
            if (seg.type === 'text') return <span key={i}>{seg.content}</span>
            const colors = LABEL_COLORS[seg.entity.label] || DEFAULT_COLOR
            return (
              <span
                key={i}
                onClick={e => handleEntityClick(e, seg.entity)}
                className={`border rounded px-1 py-0.5 mx-0.5 cursor-pointer hover:opacity-80 transition-opacity ${colors}`}
              >
                {seg.content}
                <span
                  className="ml-1 text-xs opacity-60 font-medium"
                  aria-hidden="true"
                  data-label="true"
                  style={{ userSelect: 'none' }}>
                  {seg.entity.label}
                </span>
              </span>
            )
          })}
        </div>

        {entities.length > 0 && (
          <p className="text-xs text-gray-500">
            {entities.length} entiteta · Klikni entitet za izmjenu ili brisanje · Selektiraj tekst za dodavanje
          </p>
        )}
        {entities.length === 0 && (
          <p className="text-xs text-gray-400">Selektiraj tekst mišem za dodavanje entiteta</p>
        )}

        <button
          onClick={() => onSave(entities)}
          disabled={entities.length === 0}
          className="px-5 py-2 bg-green-700 text-white text-sm font-medium rounded-lg hover:bg-green-800 disabled:opacity-40 transition-colors"
        >
          Save to Label Studio
        </button>
      </div>

      {/* Menu renderiran izvan glavnog diva da onClick ne propagira */}
      {menu && (
        <>
          {/* overlay koji zatvara menu klikom van */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setMenu(null)}
          />
          <div
            className="fixed z-50 bg-white border border-gray-200 rounded-lg shadow-lg p-2 w-64"
            style={{
              left: Math.min(menu.x, window.innerWidth - 280),
              top: Math.min(menu.y, window.innerHeight - 280),
            }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-xs font-medium text-gray-700 truncate max-w-40">
                {menu.type === 'edit' ? `"${menu.span}"` : `Dodaj: "${menu.span}"`}
              </span>
              {menu.type === 'edit' && (
                <button
                  onClick={deleteEntity}
                  className="text-xs text-red-600 hover:text-red-800 font-medium ml-2 shrink-0"
                >
                  Obriši
                </button>
              )}
            </div>
            <div className="max-h-56 overflow-y-auto space-y-0.5">
              {LABELS.map(label => {
                const colors = LABEL_COLORS[label] || DEFAULT_COLOR
                const isActive = menu.type === 'edit' && menu.currentLabel === label
                return (
                  <button
                    key={label}
                    onClick={() => menu.type === 'new' ? addEntity(label) : changeLabel(label)}
                    className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors ${
                      isActive ? colors + ' font-medium' : 'hover:bg-gray-50'
                    }`}
                  >
                    <span className={`inline-block w-2 h-2 rounded-full mr-2 border ${colors}`} />
                    {label}
                  </button>
                )
              })}
            </div>
          </div>
        </>
      )}
    </>
  )
}