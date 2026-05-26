# Septentrion Aperture

TRL-3 proof-of-concept for IDEaS CFP6 Ch13 — Arctic Maritime Domain Awareness.
Cross-modal SAR + AIS fusion: detection (E1), SAR–AIS correlation (E2),
confidence/darkness score (E3), entity-resolution graph stub (E4).

## Quickstart

```bash
uv sync                 # base deps (no torch) — runs the fusion layer
uv run septentrion run-skeleton   # synthetic end-to-end smoke run
uv run pytest           # fast test suite
```

Add the detector's PyTorch dependency only when needed:
`uv sync --extra cpu` (any platform) or `uv sync --extra cu128` (Linux GPU).
