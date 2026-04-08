"""add leads clean

Revision ID: 40bbe871d583
Revises: 89da84e06046
Create Date: 2026-04-07 13:21:22.832414

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '40bbe871d583'
down_revision: Union[str, Sequence[str], None] = '89da84e06046'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        'leads',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('phone_number', sa.String(20), nullable=False),
        sa.Column('transcript', sa.Text(), nullable=True),
        sa.Column('lead_score', sa.Enum('HOT','WARM','COLD','UNSCORED', name='leadscoreenum')),
        sa.Column('summary', sa.Text()),
        sa.Column('status', sa.Enum('COMPLETED','PENDING_RECALL','RECALLED','FAILED', name='callstatusenum')),
        sa.Column('recall_timestamp', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True))
    )
    # ### end Alembic commands ###


def downgrade():
    op.drop_table('leads')
    