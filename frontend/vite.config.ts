import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API + SSE to the FastAPI backend during dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/datasets": "http://localhost:8000",
      "/jobs": "http://localhost:8000",
      "/system": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
