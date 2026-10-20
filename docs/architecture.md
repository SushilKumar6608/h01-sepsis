# H01 v2 - Architecture

## Overview

H01 v2 adds a clinical-data integration layer on top of the Phase 1
LSTM Autoencoder sepsis early-warning model (AUC-ROC 0.926 on Set A).
It accepts observations from two source formats (HL7v2 and FHIR R4),
normalises them to LOINC-coded canonical form, validates and quarantines
bad data, and feeds the model via the same preprocessing pipeline used
during training.

```
Hospital A (HL7v2 ORU^R01)
        |
        v
  hl7v2_parser.py          -- parse raw HL7v2 bytes -> ParsedMessage
        |
        v
  mapper.py + code_map.yaml  -- local code -> LOINC, unit check
        |
        +---[quarantine]---> quarantine_items (SQLite / PostgreSQL)
        |
        v
  validator.py             -- range checks, required fields
        |
        v
  canonical store          -- normalised observations (LOINC, UCUM units)
        |
        v
  src/preprocessing.py     -- z-score norm (mean/std from training set A)
        |
        v
  src/model.py             -- LSTM Autoencoder, trained on PhysioNet CinC 2019
        |
        v
  RiskAssessment (FHIR R4) -- sepsis early-warning score


Hospital B (FHIR R4 Bundle / Observation)
        |
        v
  fhir_parser.py           -- parse FHIR R4 dict -> ParsedMessage (same type)
        |
        +-----> same pipeline as above
```

## Components

### integration/ingest/

- **hl7v2_parser.py** - Parses raw HL7v2 ORU^R01 messages into `ParsedMessage`
  / `ParsedObservation` dataclasses. Handles custom Z-segments, trailing
  whitespace, missing unit fields, and empty OBX values (quirks injected by
  the synthetic generator).
- **api.py** - FastAPI application exposing `/ingest/hl7v2` and
  `/ingest/fhir` POST endpoints. Returns a JSON summary of accepted,
  quarantined, and skipped observations.
- **replay.py** - `/quarantine/replay` endpoint: re-processes a batch of
  quarantine IDs through the mapper and validator. Used when a new mapping
  version is deployed and we want to recover previously rejected observations.

### integration/fhir/

- **fhir_parser.py** - Parses FHIR R4 Bundle or single Observation resource
  dicts into the same `ParsedMessage` type. LOINC codes are used directly as
  `local_code`; non-LOINC codes pass through and are quarantined as
  UNMAPPED_CODE by the mapper. Only `valueQuantity` (numeric) observations
  are extracted.

### integration/canonical/

- **code_map.yaml** - Single source of truth for the local-code -> LOINC
  mapping. Versioned via the `mapping_registry` table so each quarantine row
  records which mapping version produced it.
- **mapping_registry.py** - ORM model for the `mapping_registry` table.
  Tracks yaml SHA-256 and activation/deprecation timestamps per version.
- **mapper.py** - Applies code_map.yaml to a `ParsedMessage`. Observations
  with unknown local codes are quarantined as UNMAPPED_CODE.
- **validator.py** - Range and required-field validation. Out-of-range
  observations become OUT_OF_RANGE quarantine entries; missing required
  observations become MISSING_REQUIRED entries.
- **quarantine.py** - ORM model (`quarantine_items`) and helper functions for
  persisting and replaying quarantine entries.
- **db.py** - SQLAlchemy engine, session factory, shared `Base`, and
  `init_db()`. DATABASE_URL env var selects SQLite (dev/test) or PostgreSQL
  (production).
- **parity.py** - Cross-format parity checker: compares a HL7v2-sourced
  `ParsedMessage` against a FHIR-sourced one for the same encounter and
  reports matched, mismatched, and missing observations.

### integration/simulate/

- **hl7v2_generator.py** - Reads PhysioNet CinC 2019 Set A .psv files and
  emits one ORU^R01 message per ICU hour with at least one non-NaN value.
  Used to generate the test corpus; not part of the production path.
- **hospital_profiles.yaml** - Local code names and units for Hospital A.
  Must stay in sync with code_map.yaml (enforced by the end-to-end test
  in integration/tests/test_e2e_generator_to_mapper.py).

### integration/mapping/

- **features.yaml** - Canonical feature list: LOINC codes, UCUM units,
  category (vital / lab), global medians for imputation, and dataset
  provenance metadata.

## Data flow - unit handling

Source units in HL7v2 arrive as UCUM strings (e.g. `/min`, `mm[Hg]`,
`Cel`, `mg/dL`). The mapper checks them against the expected unit in
code_map.yaml and raises a WARNING-level quarantine entry if they differ
but does not reject the observation. The FHIR parity checker reports unit
differences as warnings on `ObservationPair.unit_warning`, also
non-blocking.

Unit conversion (Fahrenheit -> Celsius, mmol/L -> mg/dL for glucose) is
the responsibility of the sending system for Phase 1-3; conversion helpers
will be added in Phase 5.

## Database

Two tables are created by `init_db()` / the Alembic migrations:

```
mapping_registry
  version (PK)  description  yaml_sha256  activated_at  deprecated_at

quarantine_items
  id (PK)  control_id  patient_id  encounter_id
  local_code  loinc_code  value  unit
  reason_code  reason_detail  severity
  raw_sha256  mapping_version (FK -> mapping_registry.version)
  created_at  replayed_at  replayed_ok
```

For production, set `DATABASE_URL=postgresql+psycopg2://user:pass@host/h01v2`
and run `alembic upgrade head` from the project root.

## Test coverage

| Test file | Phase | Tests | What is covered |
|-----------|-------|-------|----------------|
| test_phase1.py | 1 | 14 | HL7v2 parser: parsing, quirks, idempotency |
| test_phase2.py | 2 | 17 | Mapper, validator, quarantine, replay API |
| test_phase3.py | 3 | 14 | FHIR parser, parity checker |
| test_e2e_generator_to_mapper.py | bugfix | 3 | Generator-to-mapper pipeline; local code alignment |

## Dataset

PhysioNet / Computing in Cardiology Challenge 2019 (Set A + Set B).
URL: https://physionet.org/content/challenge-2019/

- Set A: 20,336 patients, 8.8% sepsis
- Set B: 20,000 patients, 5.7% sepsis
- Features: 14 (7 vitals + 7 labs)
- Model AUC-ROC: 0.926 (Set A), 0.762 (Set B)

All data used by the integration layer is synthetic (generated from the
challenge dataset). No real patient data is processed.
