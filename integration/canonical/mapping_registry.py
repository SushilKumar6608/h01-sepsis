"""
Mapping registry - versioned table for code_map.yaml versions.

Tracks which version of code_map.yaml was active when each batch of
messages was processed.  Stored on quarantine_items.mapping_version so
we can later filter "quarantined under old mapping, potentially fixed
by new mapping" and replay only those rows.

Table: mapping_registry
  - version        : PK - semantic version string, e.g. "1.0.0"
  - description    : what changed in this version
  - yaml_sha256    : SHA-256 of the raw code_map.yaml bytes at that version
  - activated_at   : when this version was first loaded
  - deprecated_at  : NULL while active; set when a newer version is loaded

Usage
-----
    from integration.canonical.mapping_registry import get_or_create_version
    version = get_or_create_version(session, "1.0.0", yaml_sha256=sha)
    # then pass version.version to store_quarantine_items(mapping_version=...)
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import pathlib
from typing import Optional

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, Session

from integration.canonical.db import Base

logger = logging.getLogger(__name__)

_CODE_MAP_PATH = pathlib.Path(__file__).resolve().parent / "code_map.yaml"


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MappingRegistryEntry(Base):
    """One row per distinct version of code_map.yaml that has been loaded."""
    __tablename__ = "mapping_registry"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    yaml_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    activated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    deprecated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    @property
    def is_active(self) -> bool:
        return self.deprecated_at is None


# ---- helper functions ----------------------------------------------------------

def get_or_create_version(
    session: Session,
    version: str,
    *,
    yaml_sha256: Optional[str] = None,
    description: str = "",
) -> MappingRegistryEntry:
    """Return an existing registry entry or create one.

    Parameters
    ----------
    session : Session
    version : str
        Semantic version string, e.g. "1.0.0".
    yaml_sha256 : str, optional
        SHA-256 of the code_map.yaml bytes.  If omitted, computed from the
        file on disk at integration/canonical/code_map.yaml.
    description : str
        Human-readable note about what changed in this version.

    Returns
    -------
    MappingRegistryEntry
    """
    existing = session.get(MappingRegistryEntry, version)
    if existing is not None:
        return existing

    if yaml_sha256 is None:
        yaml_sha256 = _sha256_file(_CODE_MAP_PATH)

    entry = MappingRegistryEntry(
        version=version,
        description=description,
        yaml_sha256=yaml_sha256,
    )
    session.add(entry)
    session.flush()
    logger.info("Registered mapping version %s (sha256=%s...)", version, yaml_sha256[:12])
    return entry


def deprecate_version(
    session: Session,
    version: str,
) -> bool:
    """Mark a version as deprecated (no longer active).

    Returns True if the version was found and updated, False if not found.
    """
    entry = session.get(MappingRegistryEntry, version)
    if entry is None:
        return False
    if entry.deprecated_at is None:
        entry.deprecated_at = datetime.datetime.now(tz=datetime.timezone.utc)
        session.flush()
        logger.info("Deprecated mapping version %s", version)
    return True
