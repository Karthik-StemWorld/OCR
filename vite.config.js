import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/OCR/',
  server: {
    open: true, // Automatically open browser when dev server starts
    proxy: {
      '/api': {
        target: 'api.stemverse.app/OCR/api',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});