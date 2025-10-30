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
      waveColor: '#8b5cf6',
      progressColor: '#4f46e5',
      barWidth: 2,
      height: 90,
      responsive: true,
      cursorColor: '#fff',
      normalize: true,
    })
    wavesurferRef.current = ws
    ws.load(audioUrl)
    return () => ws.destroy()
  }, [audioUrl])

  return (
    <section className="glass p-6 sm:p-8">
      <div
        className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center cursor-pointer hover:border-neonPurple transition"
        onDrop={onDrop}
        onDragOver={(e)=>e.preventDefault()}
        onClick={()=>document.getElementById('file-input').click()}
      >
        <input id="file-input" type="file" accept="audio/*" className="hidden" onChange={onChange} />
        <div className="flex flex-col items-center gap-3">
          <div className="text-3xl text-neonPurple"><FaUpload /></div>
          <p className="text-white/80">Drag and drop your audio file here, or click to browse</p>
          {file && <p className="text-sm text-white/60">Selected: {file.name}</p>}
        </div>
      </div>

      <div className="mt-6">
        <div ref={containerRef} className="rounded-lg overflow-hidden bg-black/30" />
      </div>

      <div className="mt-6 flex justify-center">
        <button onClick={onPredict} disabled={!file} className="btn-glow glow disabled:opacity-50">
          Predict
        </button>
      </div>
    </section>
  )
}
