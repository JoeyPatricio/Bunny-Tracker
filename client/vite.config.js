import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
      '/recordings': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
      // The client fetches the trained classifier and the self-hosted MobileNet
      // backbone from /model; without this, inference 404s under `npm run dev`.
      '/model': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      }
    }
  }
})
