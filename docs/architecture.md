# H01 v2 — Architecture

> Generated placeholder. Will be filled after Phase 4.

## Overview

```
Hospital A (HL7v2)  ──┐
                       ├──▶  FastAPI ingestion  ──▶  Canonical DB (PostgreSQL)  ──▶  LSTM AE  ──▶  RiskAssessment (FHIR)
Hospital B (FHIR R4) ──┘
```

See the build plan for the full phase-by-phase description.
