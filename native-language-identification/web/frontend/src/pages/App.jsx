import React, { useState } from 'react'
import Navbar from '../ui/Navbar.jsx'
import UploadCard from '../ui/UploadCard.jsx'
import Results from '../ui/Results.jsx'
import Loader from '../ui/Loader.jsx'
import { predictAccent } from '../utils/api.js'

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const onPredict = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await predictAccent(file, 5)
      setResult(res)
    } catch (e) {
      setError(e?.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen text-white font-sans">
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 py-10 space-y-8">
        <UploadCard file={file} setFile={setFile} onPredict={onPredict} />

        {loading && (
          <div className="glass p-8 flex flex-col items-center gap-4">
            <Loader />
            <p className="text-lg text-white/80">Analyzing your accent... please wait 🎧</p>
          </div>
        )}

        {error && (
          <div className="glass p-4 border-red-400/50 text-red-300">{error}</div>
        )}

        {result && (
          <Results data={result} />
        )}
      </main>
      <footer className="text-center py-8 text-white/70">
        Built with ❤️ using React + HuBERT
      </footer>
    </div>
  )
}
