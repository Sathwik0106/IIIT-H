import React from 'react'

export default function Loader(){
  return (
    <div className="flex items-center justify-center">
      <div className="relative w-14 h-14">
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-neonBlue animate-spin"></div>
        <div className="absolute inset-1 rounded-full border-4 border-transparent border-t-neonPurple animate-spin" style={{animationDuration:'1.2s'}}></div>
      </div>
    </div>
  )
}
