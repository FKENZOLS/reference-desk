import path from "node:path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "lucide-react": path.resolve(__dirname, "./src/shims/lucide-react.tsx"),
      "sonner": path.resolve(__dirname, "./src/shims/sonner.tsx"),
      "class-variance-authority": path.resolve(__dirname, "./src/shims/cva.ts"),
      "clsx": path.resolve(__dirname, "./src/shims/clsx.ts"),
      "tailwind-merge": path.resolve(__dirname, "./src/shims/tailwind-merge.ts"),
      "@radix-ui/react-slot": path.resolve(__dirname, "./src/shims/slot.tsx"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/documents/api": "http://127.0.0.1:7860",
      "/workspace/api": "http://127.0.0.1:7860",
      "/quality/api": "http://127.0.0.1:7860",
      "/quality/export": "http://127.0.0.1:7860",
      "/workspace/export": "http://127.0.0.1:7860",
      "/viewer": "http://127.0.0.1:7860",
      "/sources": "http://127.0.0.1:7860"
    },
  },
})
