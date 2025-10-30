import React from 'react'
import { SiAudiomack } from 'react-icons/si'

export default function Navbar(){
  return (
    <header className="sticky top-0 z-10 bg-black/30 backdrop-blur-md border-b border-white/10">
      <div className="max-w-6xl mx-auto px-4 py-5 flex items-center gap-3">
        <div className="text-3xl text-neonCyan drop-shadow-lg"><SiAudiomack /></div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-wide bg-gradient-to-r from-neonCyan via-sky-400 to-neonPurple bg-clip-text text-transparent">
          HuBERT Accent Identifier
        </h1>
      </div>
    </header>
  )
}
