# Validated experiment artifact — 2026-09-01

This directory identifies the publication-grade gMART restriction experiment
run `validated_20260901_001700`.

The Git tree contains the compact scientific record:

- raw `results.jsonl` files, manifests and controller logs, excluding generated
  `layers/` directories;
- the expert and synthetic CSV datasets;
- cluster-bootstrap statistics, task-aware evaluation reports and the Russian
  scientific conclusions.

The complete data package, including all generated GeoJSON layers, the public
reference download and the replayed Urban API data, is attached to the GitHub
release:

<https://github.com/IDUclub/gMART/releases/tag/experiments-validated-20260901>

Archive name: `gmart-validated-20260901.tar.xz`

## Verify and restore

Verify the archive against `SHA256SUMS` before extracting it. On Linux/macOS:

```bash
sha256sum -c benchmarks/artifacts/validated_20260901_001700/SHA256SUMS
tar -xJf gmart-validated-20260901.tar.xz -C /path/to/gMART
```

On PowerShell:

```powershell
Get-FileHash .\gmart-validated-20260901.tar.xz -Algorithm SHA256
tar -xJf .\gmart-validated-20260901.tar.xz -C C:\path\to\gMART
```

The archive stores repository-relative paths and can therefore be extracted at
the repository root.

## Contents and limitations

- 18/18 full matrix cells; 1,755 result rows.
- Expert set: 95 human-authored tasks; synthetic robustness: 110 rows/model.
- Public reference data only. Forty-five private links require
  `YANDEX_OAUTH_TOKEN` and are not present.
- `territories.json` is unavailable and is not present.
- Geometry metrics are not reported: the conservative evaluator has only 12/95
  unambiguous reference matches and all three model runs hit the same GEOS
  topology error. See the committed geometry status report.
