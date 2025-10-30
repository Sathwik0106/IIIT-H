import React, { useState } from 'react'
import UploadCard from '../ui/UploadCard.jsx'
import Results from '../ui/Results.jsx'
import Loader from '../ui/Loader.jsx'
import { predictAccent } from '../utils/api.js'
import { FiHeadphones } from 'react-icons/fi'
import { TbWaveSine } from 'react-icons/tb'

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
    <div className="min-h-screen text-white font-sans bg-gradient-to-b from-[#0f0f1a] to-[#1a1a2e]">
      <header className="max-w-5xl mx-auto px-4 pt-10 pb-2">
        <div className="flex items-center gap-3">
          <span className="text-neonCyan text-3xl drop-shadow-lg"><TbWaveSine /></span>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-wide bg-gradient-to-r from-neonCyan via-sky-400 to-neonPurple bg-clip-text text-transparent">
            HuBERT Accent Identifier
          </h1>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-6 space-y-8">
        <section className="panel">
          <h2 className="panel-title">Upload Audio</h2>
          <UploadCard file={file} setFile={setFile} onPredict={onPredict} />

          {loading && (
            <div className="mt-6 flex flex-col items-center gap-3">
              <Loader />
              <p className="text-white/80 flex items-center gap-2">
                Analyzing your accent... please wait <FiHeadphones className="inline" />
              </p>
            </div>
          )}

          {error && (
            <div className="mt-4 glass p-3 border border-red-400/40 text-red-300">{error}</div>
          )}
        </section>

        {result && <Results data={result} />}
      </main>

      <footer className="text-center py-10 text-white/80">
        Built with <span className="text-pink-400">❤️</span> using <span className="font-semibold">React</span> + <span className="font-semibold">HuBERT</span>
      </footer>
    </div>
  )
}
