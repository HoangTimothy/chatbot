import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/auth": "http://localhost:8000",
      "/workspaces": "http://localhost:8000",
      "/documents": "http://localhost:8000",
      "/chat": "http://localhost:8000",
      "/admin": "http://localhost:8000",
    },
  },
});
