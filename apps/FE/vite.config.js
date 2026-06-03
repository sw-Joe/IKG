// apps/FE/vite.config.js
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    react({
      // 구형 esbuild 플래그 간섭을 원천 차단하고 OxC/Babel 최신 명세 수렴
      babel: {
        plugins: []
      }
    })
  ],
  server: {
    port: 5173,
    host: false, // 외부 가상 장비나 모바일 테스트 세션 연동망 개방용 가드레일
    strictPort: true,
  },
  // Rolldown 과도기 옵션 경고 제거
  optimizeDeps: {
    esbuildOptions: {
      target: 'es2022'
    }
  }
});