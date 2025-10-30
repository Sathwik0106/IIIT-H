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
    <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <motion.div
        initial={{opacity:0, y:20}}
        animate={{opacity:1, y:0}}
        transition={{duration:0.4}}
        className="glass p-6"
      >
        <h3 className="text-lg font-bold mb-2">Detected Language</h3>
        <p className="text-3xl font-extrabold">{detected_language}</p>
        <p className="text-white/70">Confidence: {(confidence*100).toFixed(2)}%</p>
      </motion.div>

      <motion.div
        initial={{opacity:0, y:20}}
        animate={{opacity:1, y:0}}
        transition={{duration:0.5, delay:0.1}}
        className="glass p-6"
      >
        <h3 className="text-lg font-bold mb-2">Cuisine Recommendations</h3>
        <ul className="list-disc ml-5 space-y-1">
          {cuisine_recommendations.slice(0,5).map((c, i) => (
            <li key={i} className="text-white/90">{c}</li>
          ))}
        </ul>
      </motion.div>

      <motion.div
        initial={{opacity:0, y:20}}
        animate={{opacity:1, y:0}}
        transition={{duration:0.6, delay:0.2}}
        className="glass p-6"
      >
        <h3 className="text-lg font-bold mb-2">Confidence Scores</h3>
        <ConfidenceBars top={top} />
      </motion.div>
    </section>
  )
}
