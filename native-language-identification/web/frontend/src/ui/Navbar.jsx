import React from 'react'
import { SiAudiomack } from 'react-icons/si'

export default function Navbar(){
  return (
    <header className="sticky top-0 z-10 bg-black/30 backdrop-blur-md border-b border-white/10">
      <div className="max-w-6xl mx-auto px-4 py-4 flex items-center gap-3">
        <div className="text-neonBlue text-2xl"><SiAudiomack /></div>
        <h1 className="text-xl sm:text-2xl font-extrabold tracking-wide">
          HuBERT Accent Identifier
        </h1>
      </div>
    </header>
  )
}
