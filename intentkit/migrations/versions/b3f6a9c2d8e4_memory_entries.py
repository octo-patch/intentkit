"""memory entries and per-topic summaries (idempotent)

Long-term memory moves from one merged note per scope row (``memories``)
to append-only entries (``memory_entries``) plus one synthesized summary
per topic (``memory_summaries``).

Additive: the two new tables only. ``memories`` is left exactly as it is —
nothing on the run path reads it any more, and
``scripts/import_legacy_memories.py`` splits its notes into entries so a
bad split can be redone. The table is dropped in a later release.

Revision ID: b3f6a9c2d8e4
Revises: a4e7c2f9b6d3
Create Date: 2026-09-06 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3f6a9c2d8e4"
down_revision: str | Sequence[str] | None = "a4e7c2f9b6d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the entry and summary tables (idempotent)."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            id VARCHAR PRIMARY KEY,
            agent_id VARCHAR NOT NULL,
            scope VARCHAR NOT NULL,
            scope_key VARCHAR NOT NULL,
            topic VARCHAR NOT NULL,
            claim TEXT NOT NULL,
            evidence TEXT NOT NULL DEFAULT '',
            user_stated BOOLEAN NOT NULL DEFAULT false,
            status VARCHAR NOT NULL DEFAULT 'active',
            source_user VARCHAR NOT NULL DEFAULT '',
            source_chat VARCHAR NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_entries_topic
        ON memory_entries (agent_id, scope, scope_key, topic, status, created_at)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_summaries (
            id VARCHAR PRIMARY KEY,
            agent_id VARCHAR NOT NULL,
            scope VARCHAR NOT NULL,
            scope_key VARCHAR NOT NULL,
            topic VARCHAR NOT NULL,
            constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
            summary TEXT NOT NULL DEFAULT '',
            open_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
            synthesized_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_memory_summaries_topic
        ON memory_summaries (agent_id, scope, scope_key, topic)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_summaries_scope_key
        ON memory_summaries (scope, scope_key)
        """
    )


def downgrade() -> None:
    """Drop the entry and summary tables; ``memories`` was never touched.

    Irreversible for anything recorded after the cutover: entries are the
    append-only history and nothing merges them back into ``memories``.
    """
    op.execute("DROP TABLE IF EXISTS memory_summaries")
    op.execute("DROP TABLE IF EXISTS memory_entries")
