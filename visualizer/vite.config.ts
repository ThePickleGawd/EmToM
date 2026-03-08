import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";
import dynamicDataPlugin from "./vite-data-plugin";

export default defineConfig({
  plugins: [
    react(),
    dynamicDataPlugin(),
    {
      name: "serve-outputs",
      configureServer(server) {
        server.middlewares.use("/outputs", (req, res, next) => {
          const urlPath = decodeURIComponent(req.url!).replace(/^\//, "");
          const filePath = path.join(__dirname, "..", "outputs", urlPath);
          if (fs.existsSync(filePath)) {
            res.setHeader(
              "Content-Type",
              filePath.endsWith(".png")
                ? "image/png"
                : filePath.endsWith(".json")
                  ? "application/json"
                  : "application/octet-stream",
            );
            fs.createReadStream(filePath).pipe(res);
          } else {
            res.statusCode = 404;
            res.end("Not found");
          }
        });
      },
    },
  ],
  server: {
    port: 3000,
    host: true,
  },
});
