"""Team memory endpoints.

Read the memory agents maintain automatically: the team's memory (scope
``team``) and the requesting user's own memory (scope ``user``), as the
per-topic summaries the prompts render. Channel and cron memories stay
internal to their conversations and are not exposed here.

Read-only by design: memory is append-only entries synthesized into
summaries, and reaches prompts only through the agent's ``record_memory``
tool — never from web input.
"""

import logging

from fastapi import APIRouter, Depends

from intentkit.core.memory import MemorySummaryWithAgent, list_account_memories

from app.team.auth import verify_team_member

team_memory_router = APIRouter(tags=["Memory"])

logger = logging.getLogger(__name__)


@team_memory_router.get(
    "/teams/{team_id}/memories",
    operation_id="list_team_memories",
    summary="List Memories",
)
async def list_team_memories(
    auth: tuple[str, str] = Depends(verify_team_member),
) -> list[MemorySummaryWithAgent]:
    """List the team's memory and the requesting user's own memory, one row
    per agent and topic."""
    user_id, team_id = auth
    return await list_account_memories(team_id, user_id)
