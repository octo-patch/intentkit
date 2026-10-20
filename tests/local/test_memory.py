"""Tests for the local Memory API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intentkit.core.memory import MemorySummaryWithAgent
from intentkit.utils.error import IntentKitAPIError, intentkit_api_error_handler

from app.local.memory import memory_router


def _memory(**overrides) -> MemorySummaryWithAgent:
    now = datetime.now(UTC)
    data = {
        "id": "mem-1",
        "agent_id": "agent-1",
        "scope": "team",
        "scope_key": "system",
        "topic": "team_profile",
        "topic_label": "Team profile",
        "constraints": [],
        "summary": "the team sells widgets",
        "open_questions": ["still true?"],
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
    app.include_router(memory_router)
    _ = app.exception_handler(IntentKitAPIError)(intentkit_api_error_handler)
    return TestClient(app)


def test_list_memories_uses_system_account(test_client):
    with patch(
        "app.local.memory.list_account_memories",
        new=AsyncMock(return_value=[_memory()]),
    ) as mock_list:
        response = test_client.get("/memories")

    assert response.status_code == 200
    mock_list.assert_awaited_once_with("system", "system")
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "mem-1"
    assert body[0]["agent_name"] == "Agent 1"
    assert body[0]["open_questions"] == ["still true?"]


def test_memory_is_read_only(test_client):
    response = test_client.put("/memories/mem-1", json={"content": "x"})
    assert response.status_code in (404, 405)
