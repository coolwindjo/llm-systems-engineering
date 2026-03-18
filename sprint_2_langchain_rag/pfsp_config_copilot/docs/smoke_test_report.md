# Smoke Test Report

- Timestamp: `2026-03-18T17:35:19.023621+00:00`
- Model: `gpt-4o-mini`
- Overall status: `passed`

## Checks
- `PASS` app runtime dependency path: app.py resolves to components.app_runtime.run_config_copilot_app
- `PASS` copilot creation: Created ConfigurationCopilot with model gpt-4o-mini
- `PASS` sample requirement input: Sample requirement produced a structured runtime result.
- `PASS` retrieval: 3 translated queries, 3 retrieved chunks
- `PASS` extraction: {"ServiceName": "WheelSpeedBroadcast", "ID": 288, "Class": "Event", "Frequency": "10 ms", "PlayType": "Cyclic"}
- `PASS` validation response: {"status": "needs_review", "schema_valid": true, "missing_required_fields": []}

## Runtime Summary
- Sample requirement: `Add a cyclic WheelSpeedBroadcast service with id 0x120 running every 10 ms for wheel speed signals and diagnostics publication.`
- Corpus sources: `autosar_error_handling.md, iso26262_error_handling.md`
- Translated query count: `3`
- Retrieved chunk count: `3`
- Validation status: `needs_review`

## Extraction Preview
```json
{
  "ServiceName": "WheelSpeedBroadcast",
  "ID": 288,
  "Class": "Event",
  "Frequency": "10 ms",
  "PlayType": "Cyclic"
}
```
