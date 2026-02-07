"""create ai_interactions table

Revision ID: 89da84e06046
Revises: 1df7f90102dd
Create Date: 2026-02-02 15:23:19.423400
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '89da84e06046'
down_revision: Union[str, Sequence[str], None] = '1df7f90102dd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'ai_interactions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('agent_name', sa.String(length=100), nullable=False),
        sa.Column('mode', sa.String(length=50), nullable=False),
        sa.Column('project_name', sa.String(length=255), nullable=True),
        sa.Column('input_payload', sa.JSON(), nullable=False),
        sa.Column('ai_raw_response', sa.Text(), nullable=False),
        sa.Column('ai_parsed_response', sa.JSON(), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False
        )
    )


def downgrade() -> None:
    op.drop_table('ai_interactions')
