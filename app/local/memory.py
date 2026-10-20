"""Local memory endpoints.

Read the memory agents maintain automatically, as the per-topic summaries
the prompts render. Local mode is single-user: both the team and the user
are "system", so this exposes the team-scope and user-scope rows of that
account. Channel and cron memories stay internal to their conversations
and are not exposed.

Read-only by design: memory is append-only entries synthesized into
summaries, and reaches prompts only through the agent's ``record_memory``
tool — never from web input.
"""

import logging

from fastapi import APIRouter

from intentkit.core.memory import MemorySummaryWithAgent, list_account_memories

memory_router = APIRouter(tags=["Memory"])

logger = logging.getLogger(__name__)

# Local single-user mode: team and user are both "system"
LOCAL_ID = "system"


@memory_router.get(
    "/memories",
    operation_id="list_memories",
    summary="List Memories",
)
async def list_memories() -> list[MemorySummaryWithAgent]:
    """List the team-scope and user-scope memory of the local account, one
    row per agent and topic."""
    return await list_account_memories(LOCAL_ID, LOCAL_ID)
