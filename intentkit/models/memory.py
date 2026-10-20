"""Scoped long-term memory: append-only entries plus one summary per topic.

Every agent keeps memory per **scope** (whose memory it is) and, inside a
scope, per **topic** (what may be kept — the closed whitelist lives in
``intentkit.core.memory_topics``). Scopes:

- ``team``: shared by the team using the agent — keyed by the *consuming*
  team's id, so each team talking to a public agent keeps its own memory.
- ``user``: one per user talking to the agent (web/API conversations),
  keyed by the user id.
- ``channel``: one per chat on a channel platform (Telegram, Slack, Lark,
  WeChat, Discord), keyed by the chat id — shared by everyone in it.
- ``cron``: one per autonomous task, keyed by the task id — the task's only
  carry-over between runs.

``user``/``channel``/``cron`` are mutually exclusive within a conversation;
``team`` is present whenever the conversation has a consuming team.

Two tables carry the design:

- ``memory_entries`` is **append-only**: one durable claim each, with its
  evidence, who it came from and whether a person stated it. Nothing edits
  or deletes a claim; a correction is a newer entry, and the only mutation
  is the daily sweep flipping an aged finding from ``active`` to ``stale``.
  The table is therefore also the history.
- ``memory_summaries`` holds one row per topic: the topic's whole state,
  regenerated from its active entries after every write. It is a cache of
  the entries with the conflicts already resolved, and the only thing a
  prompt ever renders.

The pre-entries ``memories`` table (one merged note per scope row) has no
model here: nothing on the run path reads it, and
``scripts/import_legacy_memories.py`` reads it through a lightweight table
construct to split its notes into entries. It is dropped in a later
release once every deployment has imported.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, ClassVar

from epyxid import XID
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from sqlalchemy import Boolean, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from intentkit.config.base import Base


class MemoryScope(str, Enum):
    """Whose memory an entry belongs to."""

    TEAM = "team"
    USER = "user"
    CHANNEL = "channel"
    CRON = "cron"


MEMORY_STATUS_ACTIVE = "active"
MEMORY_STATUS_STALE = "stale"


def _timestamp_column(*, onupdate: bool = False) -> Mapped[datetime]:
    # Database-side clocks throughout: the sweep compares an entry's
    # ``updated_at`` against a summary's ``synthesized_at`` to find topics
    # whose rebuild died, and mixing an app-side clock into one of them
    # would let skew between the two hosts hide or invent a lag.
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now() if onupdate else None,
    )


class MemoryEntryTable(Base):
    """One durable fact an agent kept — append-only."""

    __tablename__: str = "memory_entries"
    __table_args__: tuple[Index, ...] = (
        # the synthesis input, the block's fallback read, and the watermark:
        # all of them read one topic's active set newest first
        Index(
            "ix_memory_entries_topic",
            "agent_id",
            "scope",
            "scope_key",
            "topic",
            "status",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(XID())
    )
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    scope: Mapped[str] = mapped_column(
        String, nullable=False, comment="team | user | channel | cron"
    )
    scope_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
        comment="team_id / user_id / channel chat_id / cron task_id",
    )
    topic: Mapped[str] = mapped_column(String, nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # what a person said outranks what the agent worked out, and never expires
    user_stated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=MEMORY_STATUS_ACTIVE, comment="active | stale"
    )
    # provenance for the operators and the import script — never read back
    # into a prompt
    source_user: Mapped[str] = mapped_column(String, nullable=False, default="")
    source_chat: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = _timestamp_column()
    updated_at: Mapped[datetime] = _timestamp_column(onupdate=True)


class MemorySummaryTable(Base):
    """One topic's synthesized state — what the prompt actually renders."""

    __tablename__: str = "memory_summaries"
    __table_args__: tuple[Index, ...] = (
        Index(
            "ix_memory_summaries_topic",
            "agent_id",
            "scope",
            "scope_key",
            "topic",
            unique=True,
        ),
        # the account page lists one scope key across every agent
        Index("ix_memory_summaries_scope_key", "scope", "scope_key"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(XID())
    )
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    scope: Mapped[str] = mapped_column(String, nullable=False)
    scope_key: Mapped[str] = mapped_column(String, nullable=False)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    constraints: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    open_questions: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    synthesized_at: Mapped[datetime] = _timestamp_column()
    created_at: Mapped[datetime] = _timestamp_column()
    updated_at: Mapped[datetime] = _timestamp_column(onupdate=True)


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class MemoryConstraint(BaseModel):
    """One thing a person told the agent, carried out of the synthesis
    verbatim enough to still be theirs."""

    date: str = ""
    text: str

    @field_validator("date", mode="before")
    @classmethod
    def _iso_date_or_blank(cls, v: Any) -> str:
        """Dates sort constraints — across topics in the block, and oldest
        first when a topic is over its cap — so a date the model wrote in
        some other shape is blanked rather than left to sort as garbage."""
        match = _ISO_DATE.match(str(v or "").strip())
        return match.group(0) if match else ""


class TopicSynthesis(BaseModel):
    """A topic's whole state, regenerated on every rebuild. Also the shape
    the synthesis model is asked to answer in.

    The list halves are read defensively on both paths: an item without
    text is nothing to render, and a shape the model — or an older writer
    of the JSONB — got wrong must not fail the rebuild, or take the prompt
    or the account page down.
    """

    constraints: list[MemoryConstraint] = Field(default_factory=list)
    summary: str = ""
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("constraints", mode="before")
    @classmethod
    def _clean_constraints(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return []
        return [
            item
            for item in v
            if (isinstance(item, MemoryConstraint) and item.text.strip())
            or (isinstance(item, dict) and str(item.get("text") or "").strip())
        ]

    @field_validator("open_questions", mode="before")
    @classmethod
    def _clean_questions(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [q for q in v if isinstance(q, str) and q.strip()]

    @field_validator("summary", mode="before")
    @classmethod
    def _summary_text(cls, v: Any) -> str:
        return v if isinstance(v, str) else ""


class MemorySummary(TopicSynthesis):
    """Read model of one summary row: a synthesis plus where it is filed."""

    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)

    id: Annotated[str, Field(description="Summary ID")]
    agent_id: Annotated[str, Field(description="Agent this memory belongs to")]
    scope: Annotated[str, Field(description="team | user | channel | cron")]
    scope_key: Annotated[str, Field(description="Key within the scope")]
    topic: Annotated[str, Field(description="Topic the summary is filed under")]
    synthesized_at: Annotated[datetime, Field(description="Last rebuild time")]
    created_at: Annotated[datetime, Field(description="Creation timestamp")]
    updated_at: Annotated[datetime, Field(description="Last update timestamp")]

    @field_serializer("synthesized_at", "created_at", "updated_at")
    @classmethod
    def serialize_datetime(cls, v: datetime) -> str:
        return v.isoformat(timespec="milliseconds")
