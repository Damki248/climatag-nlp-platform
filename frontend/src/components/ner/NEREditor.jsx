const LABEL_COLORS = {
  'Asset':                     { bg: 'bg-red-100',     text: 'text-red-800',     border: 'border-red-300' },
  'Body Part':                 { bg: 'bg-orange-100',  text: 'text-orange-800',  border: 'border-orange-300' },
  'Body of Water':             { bg: 'bg-blue-100',    text: 'text-blue-800',    border: 'border-blue-300' },
  'Chemical':                  { bg: 'bg-purple-100',  text: 'text-purple-800',  border: 'border-purple-300' },
  'Climate Model':             { bg: 'bg-emerald-100', text: 'text-emerald-800', border: 'border-emerald-400' },
  'Disease':                   { bg: 'bg-red-200',     text: 'text-red-900',     border: 'border-red-400' },
  'Ecosystem':                 { bg: 'bg-green-100',   text: 'text-green-800',   border: 'border-green-300' },
  'Energy Source':             { bg: 'bg-yellow-100',  text: 'text-yellow-800',  border: 'border-yellow-300' },
  'Field of Study':            { bg: 'bg-indigo-100',  text: 'text-indigo-800',  border: 'border-indigo-300' },
  'Geographical Feature':      { bg: 'bg-violet-100',  text: 'text-violet-800',  border: 'border-violet-300' },
  'Intellectual Artefact':     { bg: 'bg-cyan-100',    text: 'text-cyan-800',    border: 'border-cyan-300' },
  'Location':                  { bg: 'bg-cyan-100',    text: 'text-cyan-800',    border: 'border-cyan-300' },
  'Mathematical Expression':   { bg: 'bg-fuchsia-100', text: 'text-fuchsia-800', border: 'border-fuchsia-300' },
  'Measuring Device':          { bg: 'bg-lime-100',    text: 'text-lime-800',    border: 'border-lime-300' },
  'Meteorological Phenomenon': { bg: 'bg-sky-100',     text: 'text-sky-800',     border: 'border-sky-300' },
  'Method':                    { bg: 'bg-amber-100',   text: 'text-amber-800',   border: 'border-amber-300' },
  'Natural Disaster':          { bg: 'bg-red-200',     text: 'text-red-900',     border: 'border-red-400' },
  'Natural Phenomenon':        { bg: 'bg-teal-100',    text: 'text-teal-900',    border: 'border-teal-400' },
  'Organism':                  { bg: 'bg-green-100',   text: 'text-green-900',   border: 'border-green-400' },
  'Organization':              { bg: 'bg-blue-100',    text: 'text-blue-900',    border: 'border-blue-400' },
  'Other':                     { bg: 'bg-gray-100',    text: 'text-gray-600',    border: 'border-gray-300' },
  'Person':                    { bg: 'bg-violet-100',  text: 'text-violet-900',  border: 'border-violet-400' },
  'Physical Artefact':         { bg: 'bg-orange-100',  text: 'text-orange-900',  border: 'border-orange-400' },
  'Physical Phenomenon':       { bg: 'bg-teal-100',    text: 'text-teal-800',    border: 'border-teal-300' },
  'Policy':                    { bg: 'bg-orange-100',  text: 'text-orange-900',  border: 'border-orange-400' },
  'Quantity':                  { bg: 'bg-yellow-100',  text: 'text-yellow-800',  border: 'border-yellow-300' },
  'Satellite':                 { bg: 'bg-gray-100',    text: 'text-gray-700',    border: 'border-gray-300' },
  'System':                    { bg: 'bg-blue-100',    text: 'text-blue-800',    border: 'border-blue-300' },
  'Time Period':               { bg: 'bg-orange-100',  text: 'text-orange-800',  border: 'border-orange-300' },
}

const DEFAULT_COLOR = { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-300' }

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

export default function NEREditor({ text, entities }) {
  const segments = buildSegments(text, entities)
  const uniqueLabels = [...new Set(entities.map(e => e.label))]

  return (
    <div className="space-y-5">
      <div className="bg-white rounded-lg border border-gray-200 px-5 py-4 text-sm leading-8">
        {segments.map((seg, i) => {
          if (seg.type === 'text') return <span key={i}>{seg.content}</span>
          const colors = LABEL_COLORS[seg.entity.label] || DEFAULT_COLOR
          return (
            <span key={i} className="relative group inline-block">
              <span className={`${colors.bg} ${colors.text} border ${colors.border} rounded px-1 py-0.5 mx-0.5 cursor-default`}>
                {seg.content}
                <span className="ml-1 text-xs opacity-60 font-medium">{seg.entity.label}</span>
              </span>
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                {seg.entity.label} · {(seg.entity.score * 100).toFixed(1)}%
              </span>
            </span>
          )
        })}
      </div>

      <div>
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
          {entities.length} entities detected
        </p>
        <div className="flex flex-wrap gap-2">
          {uniqueLabels.map(label => {
            const colors = LABEL_COLORS[label] || DEFAULT_COLOR
            const count = entities.filter(e => e.label === label).length
            return (
              <span key={label} className={`${colors.bg} ${colors.text} border ${colors.border} text-xs px-2 py-1 rounded-full`}>
                {label} ({count})
              </span>
            )
          })}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 text-left text-xs text-gray-500 uppercase tracking-wide">
              <th className="pb-2 pr-4">Span</th>
              <th className="pb-2 pr-4">Label</th>
              <th className="pb-2">Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {entities.map((e, i) => {
              const colors = LABEL_COLORS[e.label] || DEFAULT_COLOR
              return (
                <tr key={i}>
                  <td className="py-2 pr-4 font-medium">{e.span}</td>
                  <td className="py-2 pr-4">
                    <span className={`${colors.bg} ${colors.text} text-xs px-2 py-0.5 rounded`}>
                      {e.label}
                    </span>
                  </td>
                  <td className="py-2">
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-green-600 h-1.5 rounded-full"
                          style={{ width: `${e.score * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500">{(e.score * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}