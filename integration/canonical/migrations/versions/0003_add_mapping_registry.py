"""add mapping_registry table and mapping_version FK on quarantine_items

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-20

Changes:
  1. Create mapping_registry table (PK: version string).
  2. Add mapping_version column to quarantine_items (nullable FK).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create the mapping_registry table
    op.create_table(
        "mapping_registry",
        sa.Column("version", sa.String(64), primary_key=True),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("yaml_sha256", sa.String(64), nullable=False),
        sa.Column(
            "activated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deprecated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_mapping_registry_yaml_sha256", "mapping_registry", ["yaml_sha256"], unique=True
    )

    # 2. Add mapping_version FK column to quarantine_items
    op.add_column(
        "quarantine_items",
        sa.Column(
            "mapping_version",
            sa.String(64),
            sa.ForeignKey("mapping_registry.version", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_quarantine_mapping_version", "quarantine_items", ["mapping_version"]
    )


def downgrade() -> None:
    op.drop_index("ix_quarantine_mapping_version", table_name="quarantine_items")
    op.drop_column("quarantine_items", "mapping_version")
    op.drop_index("ix_mapping_registry_yaml_sha256", table_name="mapping_registry")
    op.drop_table("mapping_registry")
