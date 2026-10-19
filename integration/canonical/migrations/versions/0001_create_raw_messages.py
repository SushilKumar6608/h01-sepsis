"""create raw_messages table

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "raw_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("control_id", sa.String(128), nullable=False),
        sa.Column("received_at", sa.DateTime(), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "control_id", name="uq_source_control_id"),
    )
    op.create_index("ix_raw_messages_source", "raw_messages", ["source"])
    op.create_index("ix_raw_messages_received_at", "raw_messages", ["received_at"])


def downgrade():
    op.drop_index("ix_raw_messages_received_at", table_name="raw_messages")
    op.drop_index("ix_raw_messages_source", table_name="raw_messages")
    op.drop_table("raw_messages")
