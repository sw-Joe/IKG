// apps/FE/vite.config.js
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: []
      }
    })
  ],
  server: {
    port: 5173,
    host: false,
    strictPort: true,
  },
  // Rolldown 과도기 옵션 경고 제거
  optimizeDeps: {
    esbuildOptions: {
      target: 'es2022'
    }
  }
});