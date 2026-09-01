# Validated restriction experiments
Run `validated_20260901_001700`. Complete full cells: 18/18. Intervals are 95% percentile bootstrap intervals over scenario clusters (10,000 resamples); repeated generations remain inside their scenario cluster.
## Expert-authored primary set
| Model | Catalog | Schema | Repeat | N | Actual success (95% CI) | Intent | Buffered entity | Counted entity | Distance | Infra failures |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:12b | base | optional | 1 | 95 | 13.7% [5.1%, 23.2%] | 100.0% (95/95) | 70.5% (67/95) | 69.5% (66/95) | 100.0% (95/95) | 0 |
| gemma3:12b | base | required | 1 | 95 | 94.7% [90.0%, 98.8%] | 100.0% (95/95) | 71.6% (68/95) | 70.5% (67/95) | 100.0% (95/95) | 0 |
| gemma3:12b | no_catalog | required | 1 | 95 | 0.0% [0.0%, 0.0%] | 31.6% (30/95) | 0.0% (0/95) | 0.0% (0/95) | 1.1% (1/95) | 0 |
| gemma4:12b | base | required | 1 | 95 | 98.9% [96.3%, 100.0%] | 100.0% (95/95) | 77.9% (74/95) | 85.3% (81/95) | 100.0% (95/95) | 0 |
| gemma4:12b | no_catalog | required | 1 | 95 | 0.0% [0.0%, 0.0%] | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0 |
| gpt-oss:20b | base | required | 1 | 95 | 94.7% [89.8%, 98.9%] | 94.7% (90/95) | 67.4% (64/95) | 66.3% (63/95) | 94.7% (90/95) | 0 |
| gpt-oss:20b | no_catalog | required | 1 | 95 | 0.0% [0.0%, 0.0%] | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0 |
| gemma4:12b | base | required | 2 | 95 | 98.9% [96.3%, 100.0%] | 100.0% (95/95) | 77.9% (74/95) | 85.3% (81/95) | 100.0% (95/95) | 0 |
| gemma4:12b | no_catalog | required | 2 | 95 | 0.0% [0.0%, 0.0%] | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0 |
| gpt-oss:20b | base | required | 2 | 95 | 94.7% [89.8%, 98.9%] | 94.7% (90/95) | 67.4% (64/95) | 66.3% (63/95) | 94.7% (90/95) | 0 |
| gpt-oss:20b | no_catalog | required | 2 | 95 | 0.0% [0.0%, 0.0%] | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0 |
| gemma4:12b | base | optional | 1 | 95 | 87.4% [76.9%, 95.6%] | 100.0% (95/95) | 77.9% (74/95) | 85.3% (81/95) | 100.0% (95/95) | 0 |
| gpt-oss:20b | base | optional | 1 | 95 | 94.7% [89.8%, 98.9%] | 94.7% (90/95) | 67.4% (64/95) | 66.3% (63/95) | 94.7% (90/95) | 0 |
| gemma4:12b | base | optional | 2 | 95 | 87.4% [76.9%, 95.6%] | 100.0% (95/95) | 77.9% (74/95) | 85.3% (81/95) | 100.0% (95/95) | 0 |
| gpt-oss:20b | base | optional | 2 | 95 | 94.7% [89.8%, 98.9%] | 94.7% (90/95) | 67.4% (64/95) | 66.3% (63/95) | 94.7% (90/95) | 0 |

## Paired effects
| Contrast | Model | Repeat | Pairs | Δ success (95% CI) | wins/losses/ties |
|---|---|---|---|---|---|
| catalog: base − no_catalog | gemma3:12b | 1 | 95 | 94.7% [89.9%, 98.8%] | 90/0/5 |
| catalog: base − no_catalog | gemma4:12b | 1 | 95 | 98.9% [96.2%, 100.0%] | 94/0/1 |
| catalog: base − no_catalog | gemma4:12b | 2 | 95 | 98.9% [96.2%, 100.0%] | 94/0/1 |
| catalog: base − no_catalog | gpt-oss:20b | 1 | 95 | 94.7% [89.9%, 98.9%] | 90/0/5 |
| catalog: base − no_catalog | gpt-oss:20b | 2 | 95 | 94.7% [89.9%, 98.9%] | 90/0/5 |
| schema: required − optional | gemma3:12b | 1 | 95 | 81.1% [71.8%, 90.2%] | 77/0/18 |
| schema: required − optional | gemma4:12b | 1 | 95 | 11.6% [3.5%, 21.8%] | 11/0/84 |
| schema: required − optional | gemma4:12b | 2 | 95 | 11.6% [3.5%, 21.8%] | 11/0/84 |
| schema: required − optional | gpt-oss:20b | 1 | 95 | 0.0% [0.0%, 0.0%] | 0/0/95 |
| schema: required − optional | gpt-oss:20b | 2 | 95 | 0.0% [0.0%, 0.0%] | 0/0/95 |
| model: Gemma 4 − GPT-OSS | paired | 1 | 95 | 4.2% [-1.2%, 9.4%] | 5/1/89 |
| model: Gemma 4 − GPT-OSS | paired | 2 | 95 | 4.2% [-1.2%, 9.4%] | 5/1/89 |

## Synthetic robustness slice (secondary evidence)
The slice is deterministic and cluster-balanced. It is not independent expert gold: the prompts were model-generated, so it can support robustness but not replace the primary set.
| Model | N | Operational success (95% CI) |
|---|---|---|
| gemma3:12b | 110 | 98.2% [95.5%, 100.0%] |
| gemma4:12b | 110 | 97.3% [94.5%, 100.0%] |
| gpt-oss:20b | 110 | 92.7% [87.3%, 97.3%] |

## Interpretation rules
- A paired effect is treated as supported only when its cluster-bootstrap interval excludes zero.
- Model ranking is not claimed from overlapping or zero-crossing paired intervals.
- Geometry/object-selection conclusions require reference GeoJSON and are reported separately; absence of those files means those claims are not computable.
- Incomplete cells, any infrastructure failures, or a changed manifest invalidate cross-cell comparison until rerun.
