import React, { useEffect, useRef, useState } from 'react'
import { FaUpload } from 'react-icons/fa'
import WaveSurfer from 'wavesurfer.js'

export default function UploadCard({ file, setFile, onPredict }){
  const containerRef = useRef(null)
  const wavesurferRef = useRef(null)
  const [audioUrl, setAudioUrl] = useState(null)

  const onDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const f = e.dataTransfer.files[0]
      setFile(f)
    }
  }

  const onChange = (e) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0])
  }

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
    }
  }, [file])

  useEffect(() => {
    if (!audioUrl || !containerRef.current) return
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy()
    }
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#22d3ee',
      progressColor: '#4f46e5',
      barWidth: 2,
      height: 100,
      responsive: true,
      cursorColor: '#fff',
      normalize: true,
    })
    wavesurferRef.current = ws
    ws.load(audioUrl)
    return () => ws.destroy()
  }, [audioUrl])

  return (
    <section>
      <div
        className="dropzone"
        onDrop={onDrop}
        onDragOver={(e)=>e.preventDefault()}
        onClick={()=>document.getElementById('file-input').click()}
      >
        <input id="file-input" type="file" accept="audio/*" className="hidden" onChange={onChange} />
        <div className="flex flex-col items-center gap-3">
          <div className="text-3xl text-neonCyan"><FaUpload /></div>
          <p className="text-white/80">Drag & drop your audio file here...</p>
          {file && <p className="text-sm text-white/60">Selected: {file.name}</p>}
        </div>
      </div>

      <div className="mt-6">
        <div className="rounded-xl overflow-hidden bg-black/30 border border-white/10 p-2">
          <div ref={containerRef} className="h-[100px]" />
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <div className="p-[2px] rounded-full bg-gradient-to-r from-neonCyan via-blue-500 to-neonPurple shadow-neon">
          <button onClick={onPredict} disabled={!file}
            className="rounded-full px-12 py-4 text-lg font-semibold text-white bg-[#0f0f1a]/80 hover:scale-[1.02] transition disabled:opacity-50">
            Predict
          </button>
        </div>
      </div>
    </section>
  )
}
