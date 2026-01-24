import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Build directly into the Python package so `begira` can serve it without an extra sync step.
  build: {
    outDir: '../src/begira/_frontend/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/healthz': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
