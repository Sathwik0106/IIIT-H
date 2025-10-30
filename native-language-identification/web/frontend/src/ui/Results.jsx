import React from 'react'
import { motion } from 'framer-motion'

function ConfidenceBars({ top }){
  return (
    <div className="space-y-3">
      {top.map((t, idx) => (
        <div key={idx} className="">
          <div className="flex justify-between text-sm text-white/80">
            <span>{t.label}</span>
            <span>{(t.probability*100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-md overflow-hidden">
            <div className="h-2 bg-gradient-to-r from-neonBlue to-neonPurple" style={{width: `${Math.max(4, t.probability*100)}%`}} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function Results({ data }){
  const { detected_language, confidence, cuisine_recommendations = [], top = [] } = data
  return (
    <section className="panel">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div
          initial={{opacity:0, y:20}}
          animate={{opacity:1, y:0}}
          transition={{duration:0.3}}
          className="glass p-6 rounded-2xl"
        >
          <h3 className="text-lg font-bold mb-2">Detected Language</h3>
          <div className="flex items-center gap-3">
            <div className="text-3xl">🇮🇳</div>
            <p className="text-3xl font-extrabold">{detected_language}</p>
          </div>
          <p className="text-white/70 mt-1">Confidence: {(confidence*100).toFixed(2)}%</p>
        </motion.div>

        <motion.div
          initial={{opacity:0, y:20}}
          animate={{opacity:1, y:0}}
          transition={{duration:0.35, delay:0.05}}
          className="glass p-6 rounded-2xl"
        >
          <h3 className="text-lg font-bold">Cuisine</h3>
          <p className="text-sm text-white/60 -mt-1 mb-2">Recommendations</p>
          <ul className="list-disc ml-5 space-y-1">
            {cuisine_recommendations.slice(0,5).map((c, i) => (
              <li key={i} className="text-white/90">{c}</li>
            ))}
          </ul>
        </motion.div>

        <motion.div
          initial={{opacity:0, y:20}}
          animate={{opacity:1, y:0}}
          transition={{duration:0.4, delay:0.1}}
          className="glass p-6 rounded-2xl"
        >
          <h3 className="text-lg font-bold mb-2">Confidence</h3>
          <ConfidenceBars top={top} />
        </motion.div>
      </div>
    </section>
  )
}
