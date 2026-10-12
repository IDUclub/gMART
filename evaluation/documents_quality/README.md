# Качество ответов агента документов

Набор для оценки ответов `/documents/qa/stream` по общей базе DVD dev-контура.
Он дополняет `documents_latency`: там измеряется задержка, здесь — правильность.

## Состав

- `harvest.py` — выгружает корпус DVD через `/library` в
  `benchmarks/data/documents_quality/corpus` (каталог не в git). Токен Keycloak
  берётся из `DVD_TOKEN` либо выдаётся сервисному клиенту `idu-dvd` по секрету
  из `DVD_CLIENT_SECRET` или из скрытого ввода.
- `candidates.py` — отбирает из корпуса фрагменты-кандидаты для вопросов (числа
  с оговорками, исключения, определения, таблицы), не больше нескольких на
  документ.
- `cases.json` — 120 вопросов классов A–H с эталонным ответом (`reference`),
  обязательными фактами (`facts`), недопустимыми утверждениями (`forbid`) и
  источниками (`sources`: документ, адрес по тексту документа, номер фрагмента
  в корпусе и дословная выдержка).
- `verify_cases.py` — сверяет каждую выдержку и каждый факт с корпусом.

## Запуск

```powershell
$env:DVD_CLIENT_SECRET = (Get-Clipboard).Trim()
uv run python evaluation/documents_quality/harvest.py --base-url http://10.32.11.90:31002 `
    --out benchmarks/data/documents_quality/corpus
uv run python evaluation/documents_quality/verify_cases.py benchmarks/data/documents_quality/corpus
uv run python evaluation/documents_latency/run_http.py --base-url http://10.32.11.90:31004 `
    --cases evaluation/documents_quality/cases.json --label quality-0.22.0 `
    --out benchmarks/data/documents_quality/quality-0.22.0.jsonl
```

Адрес в `sources.address` записан по тексту документа. У части фрагментов номер
пункта в метаданных DVD с ним не совпадает; такие кейсы помечены в `note`.
После первого прогона набор не меняется.
