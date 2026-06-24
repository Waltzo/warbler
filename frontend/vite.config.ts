import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Read ports from ../config.toml (single source of truth shared with backend).
// Minimal section-aware parser — no extra dependency. Env vars override.
function tomlPorts(): { frontendPort: number; backendPort: number } {
  let frontendPort = 5173;
  let backendPort = 8000;
  try {
    const text = readFileSync(resolve(__dirname, "..", "config.toml"), "utf-8");
    let section = "";
    for (const raw of text.split("\n")) {
      const line = raw.replace(/#.*$/, "").trim();
      const sec = line.match(/^\[(\w+)\]$/);
      if (sec) { section = sec[1]; continue; }
      const kv = line.match(/^(\w+)\s*=\s*(\d+)/);
      if (!kv) continue;
      if (section === "frontend" && kv[1] === "port") frontendPort = Number(kv[2]);
      if (section === "server" && kv[1] === "port") backendPort = Number(kv[2]);
    }
  } catch {
    // config.toml missing -> defaults
  }
  return {
    frontendPort: Number(process.env.STT_FRONTEND_PORT ?? frontendPort),
    backendPort: Number(process.env.STT_PORT ?? backendPort),
  };
}

const { frontendPort, backendPort } = tomlPorts();
const target = `http://localhost:${backendPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    port: frontendPort,
    proxy: {
      "/datasets": target,
      "/jobs": target,
      "/system": target,
      "/health": target,
    },
  },
});
