<!-- INSERT THIS SECTION INTO README.md AFTER THE EXISTING MODEL RESULTS SECTION -->

## HL7v2 / FHIR Integration Layer (H01 v2)

H01 v2 extends the base model with a clinical-data integration layer that
accepts real-world hospital messages, maps them to canonical LOINC-coded
observations, and feeds the LSTM Autoencoder.

### Supported input formats

| Format | Entry point | Notes |
|--------|------------|-------|
| HL7v2 ORU^R01 | `POST /ingest/hl7v2` | Pipe-delimited segments, custom Z-segments handled |
| FHIR R4 Bundle | `POST /ingest/fhir` | JSON; extracts `valueQuantity` Observations |
| FHIR R4 Observation | `POST /ingest/fhir` | Single resource, same endpoint |

### Mapping: local codes -> LOINC

Each hospital sends its own internal lab codes (e.g. `HR_LOC`, `RR_LOC`).
The mapper translates them to LOINC codes using
`integration/canonical/code_map.yaml`:

```
HR_LOC    ->  8867-4   (Heart rate, /min)
RR_LOC    ->  9279-1   (Respiratory rate, /min)
GLU_LOC   ->  2345-7   (Glucose serum/plasma, mg/dL)
CREA_LOC  ->  2160-0   (Creatinine, mg/dL)
...
```

LOINC codes are the universal clinical-observation standard published by
the Regenstrief Institute (https://loinc.org). They are not randomly
assigned - each code is tied to a specific analyte, specimen, method, and
unit. For example `2345-7` is specifically glucose measured in serum or
plasma and reported in mg/dL, matching the PhysioNet CinC 2019 dataset.

The mapping version is recorded in the `mapping_registry` table alongside
every quarantined row, enabling replay experiments when the mapping is
updated.

### Quarantine and replay

Observations that fail mapping or validation are stored in the
`quarantine_items` table with one of three reason codes:

- `UNMAPPED_CODE` - local code not in code_map.yaml
- `OUT_OF_RANGE` - value outside the physiological range for that LOINC code
- `MISSING_REQUIRED` - a required observation (HR, SPO2, SBP) was absent

Quarantined items can be replayed via `POST /quarantine/replay` once the
mapping or validation rule is corrected.

### Cross-format parity check

`integration/canonical/parity.py` compares a HL7v2-sourced message against
a FHIR-sourced message for the same encounter.  It reports:
- `matched` - same LOINC code, same value (within 0.1% tolerance)
- `mismatched` - same LOINC code, different value
- `missing_from_hl7` / `missing_from_fhir` - code present on only one side
- `unit_warning` - same value but different unit strings (non-blocking)

### Running the integration tests

```bat
cd D:\Egna projekt\h01-sepsis
conda activate D:\conda-envs\h01-sepsis

:: All phases
pytest integration/tests/ -v

:: Individual phases
pytest integration/tests/test_phase1.py -v   :: HL7v2 parser (14 tests)
pytest integration/tests/test_phase2.py -v   :: Mapper / quarantine (17 tests)
pytest integration/tests/test_phase3.py -v   :: FHIR parser / parity (14 tests)
pytest integration/tests/test_e2e_generator_to_mapper.py -v  :: E2E (3 tests)
```

### Architecture

See [docs/architecture.md](docs/architecture.md) for the full component
diagram, data-flow description, and database schema.
