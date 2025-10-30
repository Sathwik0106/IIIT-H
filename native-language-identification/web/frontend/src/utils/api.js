import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000'

export async function predictAccent(file, topk=5){
  const form = new FormData()
  form.append('file', file)
  form.append('topk', String(topk))
  const { data } = await axios.post(`${API_BASE}/api/predict`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
