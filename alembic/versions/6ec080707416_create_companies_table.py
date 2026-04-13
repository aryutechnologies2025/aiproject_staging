"""initial schema

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("industry", sa.String(100), nullable=False, server_default=""),
        sa.Column("language", sa.String(10), nullable=False, server_default="ta"),
        sa.Column("agent_name", sa.String(100), nullable=False, server_default="Agent"),
        sa.Column("active_script_id", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_companies_slug", "companies", ["slug"], unique=True)

    op.create_table(
        "company_scripts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("company_id", sa.String(36), sa.ForeignKey(
            "companies.id", ondelete="CASCADE", name="fk_company_scripts_company"
        ), nullable=False),
        sa.Column("version", sa.String(50), nullable=False, server_default="1.0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("steps", sa.JSON(), nullable=False),
        sa.Column("objection_responses", sa.JSON(), nullable=False),
        sa.Column("closing_hot", sa.Text(), nullable=False, server_default=""),
        sa.Column("closing_warm", sa.Text(), nullable=False, server_default=""),
        sa.Column("closing_cold", sa.Text(), nullable=False, server_default=""),
        sa.Column("system_prompt_extra", sa.Text(), nullable=False, server_default=""),
        sa.Column("uploaded_filename", sa.String(300), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_company_scripts_company_id", "company_scripts", ["company_id"])
    op.create_index("ix_company_scripts_status", "company_scripts", ["status"])

    op.create_foreign_key(
        "fk_companies_active_script",
        "companies", "company_scripts",
        ["active_script_id"], ["id"],
        use_alter=True,
    )

    op.create_table(
        "leads",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("phone", sa.String(20), nullable=False),
        sa.Column("company_id", sa.String(36), sa.ForeignKey(
            "companies.id", ondelete="CASCADE", name="fk_leads_company"
        ), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("qualification", sa.String(200), nullable=True),
        sa.Column("experience_years", sa.Integer(), nullable=True),
        sa.Column("language_preference", sa.String(10), nullable=False, server_default="ta"),
        sa.Column("call_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("last_called_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("next_call_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("scheduled_interview_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_file", sa.String(200), nullable=True),
        sa.Column("score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_leads_phone_company", "leads", ["phone", "company_id"], unique=True)
    op.create_index("ix_leads_company_id", "leads", ["company_id"])
    op.create_index("ix_leads_status", "leads", ["status"])
    op.create_index("ix_leads_next_call_at", "leads", ["next_call_at"])

    op.create_table(
        "interview_slots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("lead_id", sa.String(36), sa.ForeignKey(
            "leads.id", ondelete="CASCADE", name="fk_interview_slots_lead"
        ), nullable=False),
        sa.Column("call_id", sa.String(36), nullable=False),
        sa.Column("company_id", sa.String(36), sa.ForeignKey(
            "companies.id", ondelete="CASCADE", name="fk_interview_slots_company"
        ), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("confirmed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("calendar_event_id", sa.String(200), nullable=True),
        sa.Column("sms_sent", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_interview_slots_lead_id", "interview_slots", ["lead_id"])
    op.create_index("ix_interview_slots_company_id", "interview_slots", ["company_id"])


def downgrade() -> None:
    op.drop_table("interview_slots")
    op.drop_index("ix_leads_next_call_at", "leads")
    op.drop_index("ix_leads_status", "leads")
    op.drop_index("ix_leads_company_id", "leads")
    op.drop_index("ix_leads_phone_company", "leads")
    op.drop_table("leads")
    op.drop_constraint("fk_companies_active_script", "companies", type_="foreignkey")
    op.drop_index("ix_company_scripts_status", "company_scripts")
    op.drop_index("ix_company_scripts_company_id", "company_scripts")
    op.drop_table("company_scripts")
    op.drop_index("ix_companies_slug", "companies")
    op.drop_table("companies")