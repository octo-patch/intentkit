"""Tests for the RecordMemoryTool system tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools.base import ToolException
from pydantic import ValidationError

from intentkit.abstracts.graph import AgentContext
from intentkit.core.memory import MemoryInputError, TopicRef
from intentkit.core.system_tools.record_memory import (
    RecordMemoryInput,
    RecordMemoryTool,
)
from intentkit.models.chat import AuthorType
from intentkit.models.memory import MemoryScope


@pytest.fixture
def mock_context():
    """Fixture for a mocked web-user context (team + user scopes active)."""
    context = MagicMock(spec=AgentContext)
    context.agent_id = "test-agent-1"
    context.chat_id = "chat-1"
    context.user_id = "user-1"
    context.team_id = "team-1"
    context.entrypoint = AuthorType.WEB
    context.is_subagent = False
    context.is_own_team = True
    context.agent = MagicMock(team_id="team-owner")
    return context


@pytest.fixture
def mock_runtime(mock_context):
    with patch("intentkit.core.system_tools.base.get_runtime") as mock_get_runtime:
        mock_get_runtime.return_value.context = mock_context
        yield mock_get_runtime


@pytest.fixture
def store():
    """The store and the synthesizer factory, patched where the tool
    imports them from."""
    synthesizer = MagicMock(model_id="test-model")
    with (
        patch("intentkit.core.memory.record_memory", new_callable=AsyncMock) as record,
        patch(
            "intentkit.core.memory.try_create_memory_synthesizer",
            new=AsyncMock(return_value=synthesizer),
        ) as create,
    ):
        yield record, create, synthesizer


class TestRecordMemoryInput:
    def test_valid_input(self):
        inp = RecordMemoryInput(
            scope=MemoryScope.TEAM, topic="team_profile", claim="x", user_stated=True
        )
        assert inp.scope == MemoryScope.TEAM
        assert inp.evidence == ""

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            RecordMemoryInput(scope=MemoryScope.TEAM, topic="t", claim="x")  # pyright: ignore[reportCallIssue]

    def test_unknown_scope_rejected(self):
        with pytest.raises(ValidationError):
            RecordMemoryInput(scope="galaxy", topic="t", claim="x", user_stated=True)  # pyright: ignore[reportArgumentType]


class TestRecordMemoryTool:
    def test_tool_metadata(self):
        tool = RecordMemoryTool()
        assert tool.name == "record_memory"
        assert tool.requires_memory_scope
        assert "append-only" in tool.description

    @pytest.mark.asyncio
    async def test_records_into_the_active_scope(self, mock_runtime, store):
        record, _, synthesizer = store
        tool = RecordMemoryTool()

        result = await tool._arun(
            scope=MemoryScope.USER,
            topic="user_preferences",
            claim="prefers dark mode",
            user_stated=True,
            evidence="said so",
            tool_call_id="call-1",
        )

        record.assert_awaited_once()
        args, kwargs = record.call_args
        assert args[0] == TopicRef(
            "test-agent-1", MemoryScope.USER, "user-1", "user_preferences"
        )
        assert kwargs["claim"] == "prefers dark mode"
        assert kwargs["user_stated"] is True
        assert kwargs["evidence"] == "said so"
        assert kwargs["source_user"] == "user-1"
        assert kwargs["source_chat"] == "chat-1"
        assert kwargs["synthesizer"] is synthesizer
        assert "user memory" in result and "`user_preferences`" in result
        assert "as stated by them" in result

    @pytest.mark.asyncio
    async def test_team_scope_uses_consuming_team(self, mock_runtime, store):
        record, _, _ = store
        await RecordMemoryTool()._arun(
            scope=MemoryScope.TEAM,
            topic="hard_rules",
            claim="never post prices",
            user_stated=False,
        )
        assert record.call_args.args[0] == TopicRef(
            "test-agent-1", MemoryScope.TEAM, "team-1", "hard_rules"
        )

    @pytest.mark.asyncio
    async def test_the_rebuild_is_billed_to_the_run(self, mock_runtime, store):
        """The synthesizer's bill hook prices the call like every other
        internal tool LLM call, under this tool call's id."""
        _, create, _ = store
        tool = RecordMemoryTool()
        with patch.object(tool, "_bill_internal_llm", new_callable=AsyncMock) as bill:
            await tool._arun(
                scope=MemoryScope.TEAM,
                topic="team_profile",
                claim="we sell widgets",
                user_stated=True,
                tool_call_id="call-7",
            )
            response = MagicMock()
            await create.call_args.kwargs["bill"](response, "model-x")
        bill.assert_awaited_once_with(response, "call-7", "model-x")

    @pytest.mark.asyncio
    async def test_no_model_still_records(self, mock_runtime, store):
        """Without a synthesis model the entry lands unsummarized and the
        block's fallback renders it until the sweep heals the topic."""
        record, create, _ = store
        create.return_value = None
        await RecordMemoryTool()._arun(
            scope=MemoryScope.TEAM,
            topic="team_profile",
            claim="we sell widgets",
            user_stated=True,
        )
        assert record.call_args.kwargs["synthesizer"] is None

    @pytest.mark.asyncio
    async def test_inactive_scope_rejected(self, mock_runtime, store):
        record, _, _ = store
        with pytest.raises(
            ToolException, match="No `cron` memory in this conversation"
        ):
            await RecordMemoryTool()._arun(
                scope=MemoryScope.CRON, topic="task_setup", claim="x", user_stated=True
            )
        record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_stores_refusal_reads_back_to_the_model(
        self, mock_runtime, store
    ):
        """Topic and claim checks live in the store; its message is the
        tool's message, so the model learns which topics WERE allowed."""
        record, _, _ = store
        record.side_effect = MemoryInputError(
            "`x` is not a `team` topic. Valid ones: ..."
        )
        with pytest.raises(ToolException, match="not a `team` topic"):
            await RecordMemoryTool()._arun(
                scope=MemoryScope.TEAM, topic="x", claim="x", user_stated=True
            )

    @pytest.mark.asyncio
    async def test_subagent_refused(self, mock_runtime, mock_context, store):
        mock_context.is_subagent = True
        with pytest.raises(ToolException, match="sub-agent"):
            await RecordMemoryTool()._arun(
                scope=MemoryScope.TEAM,
                topic="team_profile",
                claim="x",
                user_stated=True,
            )

    @pytest.mark.asyncio
    async def test_raises_tool_exception_on_error(self, mock_runtime, store):
        record, _, _ = store
        record.side_effect = Exception("DB connection failed")
        with pytest.raises(ToolException, match="Failed to record memory"):
            await RecordMemoryTool()._arun(
                scope=MemoryScope.TEAM,
                topic="team_profile",
                claim="x",
                user_stated=True,
            )
