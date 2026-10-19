"""create quarantine_items table

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-19
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quarantine_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("control_id", sa.String(64), nullable=False),
        sa.Column("patient_id", sa.String(64), nullable=False),
        sa.Column("encounter_id", sa.String(64), nullable=False),
        sa.Column("local_code", sa.String(64), nullable=True),
        sa.Column("loinc_code", sa.String(32), nullable=True),
        sa.Column("value", sa.Float, nullable=True),
        sa.Column("unit", sa.String(32), nullable=True),
        sa.Column("reason_code", sa.String(64), nullable=False),
        sa.Column("reason_detail", sa.Text, nullable=False),
        sa.Column("severity", sa.String(8), nullable=False, server_default="ERROR"),
        sa.Column("raw_sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("replayed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replayed_ok", sa.Boolean, nullable=True),
    )
    op.create_index("ix_quarantine_control_id", "quarantine_items", ["control_id"])
    op.create_index("ix_quarantine_raw_sha256", "quarantine_items", ["raw_sha256"])


def downgrade() -> None:
    op.drop_index("ix_quarantine_raw_sha256", table_name="quarantine_items")
    op.drop_index("ix_quarantine_control_id", table_name="quarantine_items")
    op.drop_table("quarantine_items")
