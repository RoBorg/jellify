import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// `npm run dev:phone` serves over HTTPS on the LAN: mobile browsers only
// deliver devicemotion events to secure origins.
export default defineConfig({
  plugins: process.env.HTTPS ? [basicSsl()] : [],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
