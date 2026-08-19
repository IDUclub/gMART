import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/outfit";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";
import App from "./App";
import ErrorBoundary from "./ErrorBoundary";
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
