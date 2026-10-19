"""
Phase 1 — HL7v2 ORU^R01 message generator
==========================================
Reads PhysioNet Set A .psv files and emits one ORU^R01 message per
ICU hour that has at least one non-NaN observation.

Each message uses Hospital A's local lab codes and units (see
hospital_profiles.yaml).  Several realistic quirks are injected
to make parsing non-trivial:
  - trailing whitespace on ~10% of OBX values
  - missing units field on ~10% of OBX segments
  - a custom ZPT segment on every message
  - empty OBX value for ~5% of observations (NaN in source)

Usage
-----
Generate messages for a single patient (prints to stdout):
    python -m integration.simulate.hl7v2_generator --patient p000001.psv

Generate messages for the first N patients and write to a file:
    python -m integration.simulate.hl7v2_generator --n 10 --out data/hl7v2_sample.hl7

Generate messages for ALL Set A patients (large — use with --out):
    python -m integration.simulate.hl7v2_generator --all --out data/hl7v2_setA.hl7
"""

import argparse
import hashlib
import pathlib
import random
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_RAW_A   = REPO_ROOT / "data" / "raw" / "training_setA" / "training"
PROFILES_PATH = pathlib.Path(__file__).parent / "hospital_profiles.yaml"

# ── feature list (must match preprocessing.py) ─────────────────────────────────
FEATURES = [
    "HR", "O2Sat", "Resp", "SBP", "MAP", "Temp", "Glucose",
    "DBP", "Creatinine", "BUN", "Potassium", "WBC", "Platelets", "Hgb",
]

# Reference ranges for OBX — used for H/L/N flags (approximate, not clinical)
REF_RANGES: dict[str, tuple[float, float]] = {
    "HR": (60, 100), "O2Sat": (95, 100), "Resp": (12, 20),
    "SBP": (90, 140), "MAP": (70, 105), "Temp": (36.0, 37.5),
    "Glucose": (70, 140), "DBP": (60, 90), "Creatinine": (0.6, 1.2),
    "BUN": (7, 25), "Potassium": (3.5, 5.0), "WBC": (4.0, 11.0),
    "Platelets": (150, 400), "Hgb": (12.0, 17.5),
}

RANDOM_SEED = 42


def load_profile() -> dict:
    with open(PROFILES_PATH) as f:
        return yaml.safe_load(f)["hospital_A"]


def abnormal_flag(feature: str, value: float) -> str:
    lo, hi = REF_RANGES.get(feature, (None, None))
    if lo is None:
        return "N"
    if value < lo:
        return "L"
    if value > hi:
        return "H"
    return "N"


def make_control_id(patient_id: str, hour: int) -> str:
    """Deterministic, unique message control ID."""
    raw = f"{patient_id}_{hour:04d}"
    return hashlib.md5(raw.encode()).hexdigest()[:20].upper()


def make_timestamp(base_dt: datetime, hour_offset: int) -> str:
    """HL7v2 timestamp: YYYYMMDDHHmmss"""
    dt = base_dt + timedelta(hours=hour_offset)
    return dt.strftime("%Y%m%d%H%M%S")


def format_oru_r01(
    profile: dict,
    patient_id: str,
    encounter_id: str,
    hour: int,
    observations: list[tuple[str, float]],   # (feature_name, value) — NaN already filtered
    base_dt: datetime,
    rng: random.Random,
    quirks: list[str],
) -> str:
    """
    Build one ORU^R01 HL7v2 message as a string.
    Segments separated by \\r (carriage return) per HL7 spec.
    """
    sep = "|"
    comp = "^"
    ts = make_timestamp(base_dt, hour)
    ctrl_id = make_control_id(patient_id, hour)
    local_codes = profile["local_codes"]

    segments = []

    # MSH — message header
    segments.append(
        f"MSH|^~\\&"
        f"|{profile['sending_application']}|{profile['sending_facility']}"
        f"|{profile['receiving_application']}|{profile['receiving_facility']}"
        f"|{ts}||ORU^R01|{ctrl_id}|P|{profile['hl7_version']}"
    )

    # PID — patient identification (de-identified: only internal ID)
    segments.append(
        f"PID|1||{patient_id}^^^{profile['sending_facility']}^MR"
        f"||SYNTHETIC^PATIENT||19700101|U"
    )

    # PV1 — patient visit (ICU)
    segments.append(
        f"PV1|1|I|ICU^^^{profile['sending_facility']}|||||||||||||||{encounter_id}"
    )

    # OBR — observation request (one per message = one "panel")
    segments.append(
        f"OBR|1|||VITALS_LABS^ICU vitals and labs^L"
        f"|||{ts}"
    )

    # OBX — one per observation
    for seq, (feature, value) in enumerate(observations, start=1):
        info = local_codes[feature]
        code = info["code"]
        desc = info["description"]
        unit = info["unit"]
        flag = abnormal_flag(feature, value)

        # Quirk: missing units on ~10% of OBX segments
        if "missing_obx_units" in quirks and rng.random() < 0.10:
            unit = ""

        # Quirk: trailing whitespace on ~10% of values
        value_str = f"{value:.4f}"
        if "trailing_whitespace" in quirks and rng.random() < 0.10:
            value_str += "  "

        ref_lo, ref_hi = REF_RANGES.get(feature, ("", ""))
        ref_range = f"{ref_lo}-{ref_hi}" if ref_lo != "" else ""

        segments.append(
            f"OBX|{seq}|NM|{code}^{desc}^L"
            f"||{value_str}|{unit}|{ref_range}|{flag}|||F|||{ts}"
        )

    # Quirk: custom ZPT (patient transport) segment on every message
    if "custom_z_segment" in quirks:
        segments.append(f"ZPT|1|ICU_BED_{rng.randint(1,30):02d}|STABLE")

    return "\r".join(segments) + "\r"


def generate_messages_for_patient(
    psv_path: pathlib.Path,
    profile: dict,
    rng: random.Random,
    base_dt: datetime,
) -> list[str]:
    """
    Read one PhysioNet .psv file and return a list of HL7v2 message strings,
    one per ICU hour that has at least one non-NaN observation.
    """
    df = pd.read_csv(psv_path, sep="|")
    patient_id = psv_path.stem.upper()     # e.g. P000001
    encounter_id = f"ENC_{patient_id}"
    quirks = profile.get("quirks", [])
    messages = []

    for hour, row in df.iterrows():
        observations = []
        for feature in FEATURES:
            val = row.get(feature)
            if pd.isna(val):
                # Quirk: occasionally emit an OBX with empty value
                if "occasional_empty_value" in quirks and rng.random() < 0.05:
                    observations.append((feature, float("nan")))
                continue
            observations.append((feature, float(val)))

        if not observations:
            continue   # skip hours with no data at all

        # Remove NaN observations before formatting (they were only counted above)
        obs_clean = [(f, v) for f, v in observations if not pd.isna(v)]
        if not obs_clean:
            continue

        msg = format_oru_r01(
            profile=profile,
            patient_id=patient_id,
            encounter_id=encounter_id,
            hour=int(hour),
            observations=obs_clean,
            base_dt=base_dt,
            rng=rng,
            quirks=quirks,
        )
        messages.append(msg)

    return messages


def main():
    parser = argparse.ArgumentParser(description="HL7v2 ORU^R01 generator for H01 v2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patient", help="Single .psv filename, e.g. p000001.psv")
    group.add_argument("--n", type=int, help="Generate for first N patients")
    group.add_argument("--all", action="store_true", help="Generate for ALL Set A patients")
    parser.add_argument("--out", help="Output file path (default: stdout)")
    args = parser.parse_args()

    profile = load_profile()
    rng = random.Random(RANDOM_SEED)
    base_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    if args.patient:
        psv_files = [DATA_RAW_A / args.patient]
    elif args.n:
        psv_files = sorted(DATA_RAW_A.glob("p*.psv"))[: args.n]
    else:
        psv_files = sorted(DATA_RAW_A.glob("p*.psv"))

    all_messages = []
    for psv in psv_files:
        if not psv.exists():
            print(f"ERROR: {psv} not found", file=sys.stderr)
            sys.exit(1)
        msgs = generate_messages_for_patient(psv, profile, rng, base_dt)
        all_messages.extend(msgs)
        print(f"  {psv.name}: {len(msgs)} messages", file=sys.stderr)

    output = "\n".join(all_messages)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\nWrote {len(all_messages)} messages to {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
