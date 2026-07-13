import { useEffect, useRef } from "react";
import maplibregl, { Map } from "maplibre-gl";
import type { LayerData } from "./types";
const styles: Record<string, string> = {
  osm: "https://tiles.openfreemap.org/styles/bright",
  cartoLight: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  cartoDark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
};
export default function MapPanel({
  layers,
  basemap,
  onToggle,
  onBasemap,
}: {
  layers: LayerData[];
  basemap: string;
  onToggle: (id: string) => void;
  onBasemap: (id: string) => void;
}) {
  const node = useRef<HTMLDivElement>(null),
    map = useRef<Map | null>(null);
  useEffect(() => {
    if (!node.current || map.current) return;
    map.current = new maplibregl.Map({
      container: node.current,
      style: styles[basemap],
      center: [30.31, 59.94],
      zoom: 9,
    });
    map.current.addControl(new maplibregl.NavigationControl(), "top-right");
    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    m.setStyle(styles[basemap]);
  }, [basemap]);
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const draw = () => {
      const bounds = new maplibregl.LngLatBounds();
      layers.forEach((layer, i) => {
        const source = `gmart-${i}`;
        if (m.getLayer(`${source}-fill`)) m.removeLayer(`${source}-fill`);
        if (m.getLayer(`${source}-line`)) m.removeLayer(`${source}-line`);
        if (m.getLayer(`${source}-point`)) m.removeLayer(`${source}-point`);
        if (m.getSource(source)) m.removeSource(source);
        if (!layer.visible) return;
        m.addSource(source, { type: "geojson", data: layer.geojson });
        m.addLayer({
          id: `${source}-fill`,
          type: "fill",
          source,
          filter: ["==", ["geometry-type"], "Polygon"],
          paint: { "fill-color": layer.color, "fill-opacity": 0.24 },
        });
        m.addLayer({
          id: `${source}-line`,
          type: "line",
          source,
          paint: { "line-color": layer.color, "line-width": 2 },
        });
        m.addLayer({
          id: `${source}-point`,
          type: "circle",
          source,
          filter: ["==", ["geometry-type"], "Point"],
          paint: {
            "circle-color": layer.color,
            "circle-radius": 6,
            "circle-stroke-color": "#fff",
            "circle-stroke-width": 2,
          },
        });
      layer.geojson.features.forEach((f) =>
        walk((f.geometry as GeoJSON.Geometry & { coordinates?: unknown })?.coordinates, bounds),
      );
      });
      if (!bounds.isEmpty()) m.fitBounds(bounds, { padding: 55, maxZoom: 15 });
    };
    if (m.isStyleLoaded()) draw();
    else m.once("style.load", draw);
  }, [layers, basemap]);
  return (
    <div className="map-card">
      <div className="map-toolbar">
        <div>
          <span className="eyebrow">ГЕОДАННЫЕ</span>
          <strong>Карта результата</strong>
        </div>
        <select value={basemap} onChange={(e) => onBasemap(e.target.value)}>
          <option value="osm">OpenStreetMap</option>
          <option value="cartoLight">CARTO Light</option>
          <option value="cartoDark">CARTO Dark</option>
        </select>
      </div>
      <div ref={node} className="map" />
      <div className="layer-list">
        {layers.length ? (
          layers.map((l) => (
            <button
              className={`layer-chip ${l.visible ? "active" : ""}`}
              onClick={() => onToggle(l.id)}
              key={l.id}
            >
              <i style={{ background: l.color }} />
              {l.name}
              <small>{l.count}</small>
            </button>
          ))
        ) : (
          <span className="empty-inline">
            Слои появятся после геопространственного запроса
          </span>
        )}
      </div>
    </div>
  );
}
function walk(value: any, b: maplibregl.LngLatBounds) {
  if (!Array.isArray(value)) return;
  if (typeof value[0] === "number" && typeof value[1] === "number") {
    b.extend([value[0], value[1]]);
    return;
  }
  value.forEach((v) => walk(v, b));
}
