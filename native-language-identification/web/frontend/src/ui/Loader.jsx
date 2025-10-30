import React from 'react'

export default function Loader(){
  return (
    <div className="relative">
      <div className="w-16 h-16 rounded-full border-4 border-neonBlue/30 border-t-neonPurple animate-spin"></div>
      <div className="absolute inset-0 blur-lg rounded-full shadow-neon"></div>
    </div>
  )
}
