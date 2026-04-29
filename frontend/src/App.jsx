import { Routes, Route, NavLink } from 'react-router-dom'
import NERPage from './pages/NERPage'
import ClassifyPage from './pages/ClassifyPage'
import AnnotationPage from './pages/AnnotationPage'
import ExperimentsPage from './pages/ExperimentsPage'

export default function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-8">
        <span className="text-xl font-bold text-green-700">ClimaTag</span>
        <div className="flex gap-6">
          {[
            { to: '/', label: 'NER' },
            { to: '/classify', label: 'Classify' },
            { to: '/annotate', label: 'Annotate' },
            { to: '/experiments', label: 'Experiments' },
          ].map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${
                  isActive
                    ? 'text-green-700 border-b-2 border-green-700 pb-1'
                    : 'text-gray-500 hover:text-gray-900'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="max-w-4xl mx-auto px-6 py-10">
        <Routes>
          <Route path="/" element={<NERPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/annotate" element={<AnnotationPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
        </Routes>
      </main>
    </div>
  )
}