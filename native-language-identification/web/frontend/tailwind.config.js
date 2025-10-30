/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Poppins', 'ui-sans-serif', 'system-ui'],
      },
      colors: {
        neonBlue: '#4f46e5',
        neonPurple: '#a855f7',
        neonCyan: '#22d3ee',
      },
      boxShadow: {
        neon: '0 0 20px rgba(79,70,229,0.6), 0 0 40px rgba(168,85,247,0.4)',
        aqua: '0 0 20px rgba(34,211,238,0.5), 0 0 40px rgba(79,70,229,0.35)'
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}
