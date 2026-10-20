"""Tests for the team Memory API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intentkit.core.memory import MemorySummaryWithAgent
from intentkit.utils.error import IntentKitAPIError, intentkit_api_error_handler

from app.team.auth import verify_team_member
from app.team.memory import team_memory_router


def _memory(**overrides) -> MemorySummaryWithAgent:
    now = datetime.now(UTC)
    data = {
        "id": "mem-1",
        "agent_id": "agent-1",
        "scope": "team",
        "scope_key": "team-1",
        "topic": "team_profile",
        "topic_label": "Team profile",
        "constraints": [{"date": "2026-09-01", "text": "we sell widgets"}],
        "summary": "the team sells widgets",
        "open_questions": [],
        "synthesized_at": now,
        "created_at": now,
        "updated_at": now,
        "agent_name": "Agent 1",
        "agent_picture": None,
    }
    data.update(overrides)
    return MemorySummaryWithAgent.model_validate(data)


@pytest.fixture
def test_client():
    app = FastAPI()
    app.include_router(team_memory_router)
    _ = app.exception_handler(IntentKitAPIError)(intentkit_api_error_handler)
    # The routes take user/team from the verified auth tuple, not the path.
    app.dependency_overrides[verify_team_member] = lambda: ("user-1", "team-1")
    return TestClient(app)


def test_list_memories_scoped_to_auth(test_client):
    with patch(
        "app.team.memory.list_account_memories",
        new=AsyncMock(return_value=[_memory()]),
    ) as mock_list:
        response = test_client.get("/teams/team-1/memories")

    assert response.status_code == 200
    mock_list.assert_awaited_once_with("team-1", "user-1")
    body = response.json()
    assert len(body) == 1
    assert body[0]["agent_name"] == "Agent 1"
    assert body[0]["topic_label"] == "Team profile"
    assert body[0]["constraints"] == [{"date": "2026-09-01", "text": "we sell widgets"}]


def test_memory_is_read_only(test_client):
    """Memory reaches prompts only through record_memory, never from web
    input: there is no edit route."""
    response = test_client.put("/teams/team-1/memories/mem-1", json={"content": "x"})
    assert response.status_code in (404, 405)
