"""Tests for scoped long-term memory: scope resolution, the append-only
store, synthesis, decay, and what the prompt renders."""

import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from intentkit.abstracts.graph import AgentContext
from intentkit.config.base import Base
from intentkit.config.db import get_session
from intentkit.core import memory as memory_module
from intentkit.core.memory import (
    CLAIM_CHAR_CAP,
    CLAIM_OVERRUN_GRACE,
    SUMMARY_CHAR_CAP,
    MemoryInputError,
    MemoryScopeRef,
    TopicRef,
    compose_block,
    extract_json_object,
    list_account_memories,
    load_memory_block,
    load_memory_blocks,
    parse_synthesis,
    rebuild_topic,
    record_memory,
    render_memory_section,
    resolve_memory_scopes,
    sweep_memory,
)
from intentkit.core.memory_topics import SCOPE_TOPICS, topic_spec
from intentkit.models.chat import AuthorType
from intentkit.models.memory import (
    MEMORY_STATUS_ACTIVE,
    MEMORY_STATUS_STALE,
    MemoryConstraint,
    MemoryEntryTable,
    MemoryScope,
    MemorySummary,
    MemorySummaryTable,
    TopicSynthesis,
)


@pytest_asyncio.fixture()
async def memory_tables(db_engine):
    async with db_engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[MemoryEntryTable.__table__, MemorySummaryTable.__table__],
        )
    # The in-process TTL cache outlives the per-test tables; clear it so
    # tests never see blocks from a previous test's database.
    memory_module._block_cache.clear()
    yield
    memory_module._block_cache.clear()


class FakeSynthesizer:
    """Returns a canned synthesis and keeps every prompt it was handed."""

    def __init__(self, out: TopicSynthesis | None = None) -> None:
        self.out = out or TopicSynthesis(summary="synthesized")
        self.prompts: list[str] = []

    async def synthesize(self, prompt: str) -> TopicSynthesis:
        self.prompts.append(prompt)
        return self.out


def _make_context(**overrides) -> MagicMock:
    context = MagicMock(spec=AgentContext)
    context.agent_id = overrides.get("agent_id", "agent-1")
    context.chat_id = overrides.get("chat_id", "chat-1")
    context.user_id = overrides.get("user_id", "user-1")
    context.team_id = overrides.get("team_id", "team-1")
    context.entrypoint = overrides.get("entrypoint", AuthorType.WEB)
    context.is_subagent = overrides.get("is_subagent", False)
    context.is_own_team = overrides.get("is_own_team", True)
    return context


def _make_agent(team_id: str | None = "team-owner") -> MagicMock:
    agent = MagicMock()
    agent.team_id = team_id
    return agent


def team_ref(topic: str = "team_profile", agent_id: str = "agent-1") -> TopicRef:
    return TopicRef(agent_id, MemoryScope.TEAM, "team-1", topic)


TEAM = MemoryScopeRef(MemoryScope.TEAM, "team-1")


async def _entries() -> dict[str, str]:
    async with get_session() as db:
        rows = await db.scalars(select(MemoryEntryTable))
        return {r.claim: r.status for r in rows}


# --- scopes -----------------------------------------------------------------


class TestResolveMemoryScopes:
    def test_subagent_has_no_memory(self):
        context = _make_context(is_subagent=True)
        assert resolve_memory_scopes(_make_agent(), context) == []

    def test_web_user_gets_team_and_user(self):
        context = _make_context(entrypoint=AuthorType.WEB, user_id="user-9")
        scopes = resolve_memory_scopes(_make_agent(), context)
        assert scopes == [
            MemoryScopeRef(MemoryScope.TEAM, "team-1"),
            MemoryScopeRef(MemoryScope.USER, "user-9"),
        ]

    def test_consuming_team_wins_over_owning_team(self):
        """A public agent visited by another team loads the visitor's memory."""
        context = _make_context(team_id="team-visitor")
        scopes = resolve_memory_scopes(_make_agent(team_id="team-owner"), context)
        assert scopes[0].scope_key == "team-visitor"

    def test_own_team_falls_back_to_owner_then_system(self):
        context = _make_context(team_id=None, is_own_team=True)
        scopes = resolve_memory_scopes(_make_agent(team_id="team-owner"), context)
        assert scopes[0].scope_key == "team-owner"

        scopes = resolve_memory_scopes(_make_agent(team_id=None), context)
        assert scopes[0].scope_key == "system"

    def test_teamless_guest_gets_no_team_scope(self):
        """A guest without a team must never see the owning team's memory."""
        context = _make_context(team_id=None, is_own_team=False, user_id="user-9")
        scopes = resolve_memory_scopes(_make_agent(team_id="team-owner"), context)
        assert scopes == [MemoryScopeRef(MemoryScope.USER, "user-9")]

    def test_trigger_gets_cron_scope_keyed_by_task_id(self):
        context = _make_context(
            entrypoint=AuthorType.TRIGGER, chat_id="autonomous-task-42"
        )
        scopes = resolve_memory_scopes(_make_agent(), context)
        assert scopes == [
            MemoryScopeRef(MemoryScope.TEAM, "team-1"),
            MemoryScopeRef(MemoryScope.CRON, "task-42"),
        ]

    @pytest.mark.parametrize(
        "entrypoint",
        [
            AuthorType.TELEGRAM,
            AuthorType.SLACK,
            AuthorType.LARK,
            AuthorType.WECHAT,
            AuthorType.DISCORD,
        ],
    )
    def test_channel_entrypoints_get_channel_scope(self, entrypoint):
        """A chat on a channel platform may be a group: its memory is the
        chat's, keyed by the chat id, and never the sender's user scope."""
        context = _make_context(
            entrypoint=entrypoint, chat_id="thread-7", user_id="tg-9"
        )
        scopes = resolve_memory_scopes(_make_agent(), context)
        assert scopes == [
            MemoryScopeRef(MemoryScope.TEAM, "team-1"),
            MemoryScopeRef(MemoryScope.CHANNEL, "thread-7"),
        ]

    def test_anonymous_web_gets_team_only(self):
        context = _make_context(user_id=None)
        scopes = resolve_memory_scopes(_make_agent(), context)
        assert [s.scope for s in scopes] == [MemoryScope.TEAM]


# --- write path -------------------------------------------------------------


class TestRecord:
    @pytest.mark.asyncio
    async def test_record_appends_and_synthesizes(self, memory_tables):
        synth = FakeSynthesizer(
            TopicSynthesis(
                constraints=[
                    MemoryConstraint(date="2026-08-30", text="ship on Fridays")
                ],
                summary="the team ships weekly",
                open_questions=["is Friday still the day?"],
            )
        )
        await record_memory(
            team_ref("working_agreements"),
            claim="the team ships on Fridays",
            user_stated=True,
            evidence="Anne said so",
            synthesizer=synth,
        )

        block = await load_memory_block("agent-1", MemoryScope.TEAM, "team-1")
        assert "ship on Fridays" in block
        assert "the team ships weekly" in block
        assert "is Friday still the day?" in block
        # the synthesis saw the claim and its evidence, and was told which was which
        assert "the team ships on Fridays" in synth.prompts[0]
        assert "Anne said so" in synth.prompts[0]
        assert "STATED" in synth.prompts[0]

    @pytest.mark.asyncio
    async def test_a_correction_is_a_newer_entry(self, memory_tables):
        """Nothing is edited or deleted: both entries stay, and what the
        block says is whatever the synthesis made of them."""
        synth = FakeSynthesizer()
        ref = team_ref("team_resources")
        await record_memory(
            ref, claim="specs live in Notion", user_stated=True, synthesizer=synth
        )
        await record_memory(
            ref, claim="specs live in Drive", user_stated=True, synthesizer=synth
        )

        assert sorted(await _entries()) == [
            "specs live in Drive",
            "specs live in Notion",
        ]
        # newest first, so the synthesis reads the correction before the thing corrected
        assert synth.prompts[-1].index("Drive") < synth.prompts[-1].index("Notion")

    @pytest.mark.asyncio
    async def test_a_failed_synthesis_keeps_the_entry(self, memory_tables):
        class Broken:
            async def synthesize(self, prompt: str) -> TopicSynthesis:
                raise RuntimeError("model down")

        await record_memory(
            team_ref(), claim="we sell widgets", user_stated=True, synthesizer=Broken()
        )

        assert await _entries() == {"we sell widgets": MEMORY_STATUS_ACTIVE}
        # and the fallback keeps it visible until something rebuilds the topic
        assert "we sell widgets" in await load_memory_block(
            "agent-1", MemoryScope.TEAM, "team-1"
        )

    @pytest.mark.asyncio
    async def test_a_failed_rebuild_of_a_summarized_topic_still_shows_the_entry(
        self, memory_tables
    ):
        """The fallback is by entry, not by topic: a topic whose summary
        predates its newest entry renders that entry beside the summary,
        so "it reads back next turn" holds even when the rebuild died."""

        class Broken:
            async def synthesize(self, prompt: str) -> TopicSynthesis:
                raise RuntimeError("model down")

        ref = team_ref()
        await record_memory(
            ref,
            claim="we sell widgets",
            user_stated=False,
            synthesizer=FakeSynthesizer(),
        )
        await record_memory(
            ref, claim="and also gadgets", user_stated=False, synthesizer=Broken()
        )

        block = await load_memory_block("agent-1", MemoryScope.TEAM, "team-1")
        assert "synthesized" in block  # the summary the topic still has
        assert "and also gadgets" in block  # and the entry it does not cover
        assert "we sell widgets" not in block  # covered by that summary, not repeated

    @pytest.mark.asyncio
    async def test_a_rebuild_overtaken_midflight_writes_nothing(self, memory_tables):
        """The watermark: whoever changed the entries under a rebuild is
        rebuilding over the newer state, so the stale one must not clobber it."""
        ref = team_ref()
        await record_memory(
            ref, claim="first", user_stated=True, synthesizer=FakeSynthesizer()
        )

        class Interfering:
            async def synthesize(self, prompt: str) -> TopicSynthesis:
                await record_memory(
                    ref, claim="second", user_stated=True, synthesizer=None
                )
                return TopicSynthesis(summary="built from the OLD set")

        await rebuild_topic(ref, Interfering())
        async with get_session() as db:
            row = await db.scalar(select(MemorySummaryTable))
        assert row is not None and row.summary == "synthesized"

    @pytest.mark.asyncio
    async def test_the_store_refuses_what_the_whitelist_excludes(self, memory_tables):
        """The guarantee lives in the store — "a fact that fits no topic is
        not recorded" has to be a property of the write path — and its
        refusal names what WAS allowed, since the tool hands it straight
        back to the model."""
        with pytest.raises(MemoryInputError, match="not a `team` topic") as exc:
            await record_memory(
                team_ref("user_preferences"),  # a real topic, wrong scope
                claim="likes short answers",
                user_stated=True,
            )
        assert "`team_profile`" in str(exc.value)
        with pytest.raises(MemoryInputError, match="empty"):
            await record_memory(team_ref(), claim="   ", user_stated=True)
        with pytest.raises(MemoryInputError, match=f"longer than {CLAIM_CHAR_CAP}"):
            await record_memory(
                team_ref(),
                claim="x" * (CLAIM_CHAR_CAP + CLAIM_OVERRUN_GRACE + 1),
                user_stated=True,
            )
        assert await _entries() == {}

    @pytest.mark.asyncio
    async def test_a_claim_inside_the_tolerance_is_trimmed_to_the_cap(
        self, memory_tables
    ):
        await record_memory(
            team_ref(),
            claim="y" * (CLAIM_CHAR_CAP + CLAIM_OVERRUN_GRACE),
            user_stated=True,
        )
        assert list(await _entries()) == ["y" * CLAIM_CHAR_CAP]

    @pytest.mark.asyncio
    async def test_one_topic_cannot_crowd_out_the_others(self, memory_tables):
        """The block sheds WHOLE topics to fit its budget, so an unbounded
        summary would cost every other topic its place."""
        long = FakeSynthesizer(TopicSynthesis(summary="x" * (SUMMARY_CHAR_CAP * 3)))
        await record_memory(
            team_ref(), claim="we sell widgets", user_stated=False, synthesizer=long
        )

        async with get_session() as db:
            summary = await db.scalar(select(MemorySummaryTable.summary))
        assert summary is not None and len(summary) == SUMMARY_CHAR_CAP

    @pytest.mark.asyncio
    async def test_a_write_reaches_the_very_next_read(self, memory_tables):
        """The block cache is dropped on every write and every rebuild, so
        an agent sees its own entry on its next model call."""
        assert await load_memory_block("agent-1", MemoryScope.TEAM, "team-1") == ""
        await record_memory(
            team_ref(), claim="we sell widgets", user_stated=True, synthesizer=None
        )
        assert "we sell widgets" in await load_memory_block(
            "agent-1", MemoryScope.TEAM, "team-1"
        )
        await rebuild_topic(
            team_ref(), FakeSynthesizer(TopicSynthesis(summary="rebuilt"))
        )
        assert "rebuilt" in await load_memory_block(
            "agent-1", MemoryScope.TEAM, "team-1"
        )


# --- read path --------------------------------------------------------------


class TestRead:
    @pytest.mark.asyncio
    async def test_an_unsynthesized_topic_still_shows(self, memory_tables):
        """Bootstrap fallback — a write whose rebuild has not landed is never
        invisible, and a stated one still reads as a requirement rather
        than a finding."""
        await record_memory(
            team_ref(), claim="we sell widgets", user_stated=True, synthesizer=None
        )
        await record_memory(
            team_ref("domain_terms"),
            claim="'seat' means a subagent",
            user_stated=False,
            synthesizer=None,
        )
        block = await load_memory_block("agent-1", MemoryScope.TEAM, "team-1")
        stated, found = (
            block.index("we sell widgets"),
            block.index("'seat' means a subagent"),
        )
        assert (
            block.index("standing requirements")
            < stated
            < block.index("Worked out by you")
        )
        assert found > block.index("Worked out by you")

    @pytest.mark.asyncio
    async def test_one_scope_never_reads_another(self, memory_tables):
        await record_memory(
            TopicRef("agent-1", MemoryScope.USER, "user-1", "user_preferences"),
            claim="prefers short answers",
            user_stated=True,
            synthesizer=None,
        )
        mine = await load_memory_blocks(
            "agent-1", [TEAM, MemoryScopeRef(MemoryScope.USER, "user-1")]
        )
        assert "short answers" in mine[MemoryScope.USER]
        assert mine[MemoryScope.TEAM] == ""

        other_user = await load_memory_block("agent-1", MemoryScope.USER, "user-2")
        other_agent = await load_memory_block("agent-2", MemoryScope.USER, "user-1")
        assert other_user == "" and other_agent == ""

    @pytest.mark.asyncio
    async def test_reads_are_cached_between_model_calls(self, memory_tables):
        await record_memory(
            team_ref(), claim="we sell widgets", user_stated=True, synthesizer=None
        )
        first = await load_memory_block("agent-1", MemoryScope.TEAM, "team-1")
        with patch.object(
            memory_module, "_read_blocks", new=AsyncMock(return_value={})
        ) as read:
            second = await load_memory_block("agent-1", MemoryScope.TEAM, "team-1")
        assert second == first
        read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_write_during_a_read_is_not_hidden_by_the_cache(
        self, memory_tables
    ):
        """A read that started before a write and lands after it must not
        cache what it fetched — the write's tombstone is newer than the
        read's start, so the next call refetches and sees the entry."""
        real_read = memory_module._read_blocks

        async def read_then_write(agent_id, wanted, **kw):
            blocks = await real_read(agent_id, wanted, **kw)
            await record_memory(team_ref(), claim="landed mid-read", user_stated=True)
            return blocks

        with patch.object(memory_module, "_read_blocks", new=read_then_write):
            stale = await load_memory_block("agent-1", MemoryScope.TEAM, "team-1")
        assert stale == ""
        assert "landed mid-read" in await load_memory_block(
            "agent-1", MemoryScope.TEAM, "team-1"
        )


class TestComposeBlock:
    @staticmethod
    def _row(
        topic: str, summary: str, synthesized_at: datetime, **overrides
    ) -> MemorySummary:
        now = datetime.now(UTC)
        data: dict[str, Any] = {
            "id": f"s-{topic}",
            "agent_id": "agent-1",
            "scope": "team",
            "scope_key": "team-1",
            "topic": topic,
            "summary": summary,
            "constraints": [{"date": "2026-08-01", "text": "never post on weekends"}],
            "open_questions": ["still true?"],
            "synthesized_at": synthesized_at,
            "created_at": now,
            "updated_at": now,
        }
        data.update(overrides)
        return MemorySummary.model_validate(data)

    def test_the_budget_sheds_learnings_and_never_constraints(self):
        now = datetime.now(UTC)
        rows = [
            self._row("team_profile", "old " * 200, now - timedelta(days=9)),
            self._row("domain_terms", "new " * 200, now),
        ]
        block = compose_block(MemoryScope.TEAM, rows, char_budget=1300)  # room for one
        assert (
            "never post on weekends" in block
        )  # constraints survive whatever else goes
        assert (
            "old old" not in block
        )  # the least recently synthesized topic sheds first
        assert "new new" in block

    def test_the_budget_degrades_by_whole_items_before_it_cuts(self):
        """The shed order in full: learnings, then questions, then the
        OLDEST constraints — one whole line at a time."""
        row = self._row(
            "working_agreements",
            "a finding " * 40,
            datetime.now(UTC),
            constraints=[
                {"date": f"2026-0{m}-01", "text": f"statement from month {m} " * 8}
                for m in range(1, 9)
            ],
        )
        block = compose_block(MemoryScope.TEAM, [row], char_budget=1000)
        assert "a finding" not in block  # learnings go first
        assert "still true?" not in block  # then the questions
        assert "month 1 " not in block  # then the oldest statements
        assert "month 8 " in block  # the newest survives
        assert len(block) <= 1000
        assert block.rstrip().endswith(")")  # a whole line, not a severed one

    def test_odd_json_shapes_render_as_nothing(self):
        """A shape an older writer left behind must not take the prompt down."""
        row = self._row(
            "team_profile",
            "",
            datetime.now(UTC),
            constraints=[{"date": "x"}, "junk", {"text": "kept"}],
            open_questions=["", None, "asked?"],
        )
        block = compose_block(MemoryScope.TEAM, [row])
        assert [c.text for c in row.constraints] == ["kept"]
        assert row.open_questions == ["asked?"]
        assert "kept" in block and "asked?" in block


# --- decay ------------------------------------------------------------------


class TestSweep:
    @pytest.mark.asyncio
    async def test_the_sweep_expires_findings_and_never_statements(self, memory_tables):
        ref = team_ref("hard_rules")  # 90-day cap
        await record_memory(ref, claim="a finding", user_stated=False, synthesizer=None)
        await record_memory(
            ref, claim="a statement", user_stated=True, synthesizer=None
        )

        synth = FakeSynthesizer()

        swept = await sweep_memory(datetime.now(UTC) + timedelta(days=91), synth)

        assert swept == 1
        assert await _entries() == {
            "a finding": MEMORY_STATUS_STALE,
            "a statement": MEMORY_STATUS_ACTIVE,
        }
        assert synth.prompts and "a finding" not in synth.prompts[-1]

    @pytest.mark.asyncio
    async def test_the_sweep_reads_the_ttl_from_the_registry(self, memory_tables):
        """Caps are not stored on the row: `team_profile` is capped at a
        year and `working_agreements` at half of one, so retuning a topic
        reaches rows already written."""
        await record_memory(
            team_ref(), claim="year-long", user_stated=False, synthesizer=None
        )
        await record_memory(
            team_ref("working_agreements"),
            claim="half-year",
            user_stated=False,
            synthesizer=None,
        )

        await sweep_memory(datetime.now(UTC) + timedelta(days=200), FakeSynthesizer())
        assert await _entries() == {
            "year-long": MEMORY_STATUS_ACTIVE,
            "half-year": MEMORY_STATUS_STALE,
        }

    @pytest.mark.asyncio
    async def test_the_sweep_heals_a_rebuild_that_died(self, memory_tables):
        """A write whose rebuild failed leaves entries with no summary.
        Nothing would retrigger it until the next write to the same topic,
        so the sweep does."""
        await record_memory(
            team_ref(), claim="we sell widgets", user_stated=True, synthesizer=None
        )

        synth = FakeSynthesizer(TopicSynthesis(summary="healed"))

        assert await sweep_memory(datetime.now(UTC), synth) == 0
        assert "healed" in await load_memory_block(
            "agent-1", MemoryScope.TEAM, "team-1"
        )

    @pytest.mark.asyncio
    async def test_a_sweep_rebuilds_at_most_the_cap_and_the_rest_heal_next_time(
        self, memory_tables, monkeypatch
    ):
        monkeypatch.setattr(memory_module, "SWEEP_REBUILD_CAP", 1)
        await record_memory(team_ref(), claim="one", user_stated=True)
        await record_memory(team_ref("domain_terms"), claim="two", user_stated=True)

        synth = FakeSynthesizer()
        await sweep_memory(datetime.now(UTC), synth)
        assert len(synth.prompts) == 1
        await sweep_memory(datetime.now(UTC), synth)
        assert len(synth.prompts) == 2  # the leftover, not a repeat
        await sweep_memory(datetime.now(UTC), synth)
        assert len(synth.prompts) == 2  # nothing lagging any more

    @pytest.mark.asyncio
    async def test_the_last_entry_expiring_clears_the_summary(self, memory_tables):
        """No entries left means no model call — and the topic has to leave
        the block, not keep showing what it said before."""
        ref = team_ref("hard_rules")
        await record_memory(
            ref, claim="a finding", user_stated=False, synthesizer=FakeSynthesizer()
        )
        assert "synthesized" in await load_memory_block(
            "agent-1", MemoryScope.TEAM, "team-1"
        )

        called = FakeSynthesizer()

        await sweep_memory(datetime.now(UTC) + timedelta(days=91), called)
        assert called.prompts == []  # nothing to summarize, so nothing was paid for
        assert await load_memory_block("agent-1", MemoryScope.TEAM, "team-1") == ""


# --- the prompt and the tool must describe the same memory --------------------


class TestPromptSection:
    def test_the_prompt_offers_exactly_what_the_tool_accepts(self):
        """The whitelist is rendered in the prompt and enforced in the tool,
        in two different files. If they drift, the agent spends turns being
        refused for topics it was just told to use — so this reads the
        rendered text back and checks every scope/topic pair in it against
        the registry the tool consults."""
        scopes = [
            MemoryScope.TEAM,
            MemoryScope.CHANNEL,
            MemoryScope.USER,
            MemoryScope.CRON,
        ]
        rendered = render_memory_section({scope: "" for scope in scopes})

        offered: list[tuple[MemoryScope, str]] = []
        current: MemoryScope | None = None
        for line in rendered.splitlines():
            if match := re.fullmatch(r"- `(\w+)` \(.*\):", line):
                current = MemoryScope(match.group(1))
            elif (match := re.fullmatch(r"  - `(\w+)` — .*", line)) and current:
                offered.append((current, match.group(1)))

        assert offered, "the whitelist rendered no topics at all"
        assert {scope for scope, _ in offered} == set(scopes)
        for scope, topic in offered:
            assert topic_spec(scope, topic) is not None, (
                f"prompt offers {scope.value}.{topic}"
            )
        # and nothing the tool accepts is left unsaid, which is the other direction
        for scope in scopes:
            named = {t for s, t in offered if s == scope}
            assert named == {spec.value for spec in SCOPE_TOPICS[scope]}

    def test_blocks_render_under_their_headings_in_scope_order(self):
        rendered = render_memory_section(
            {MemoryScope.USER: "likes cats", MemoryScope.TEAM: "sells widgets"}
        )
        assert rendered.startswith("## Memory")
        assert "record_memory" in rendered
        assert "stored data" in rendered
        assert rendered.index("### Team Memory") < rendered.index("### User Memory")
        assert "### Cron Task Memory" not in rendered  # no dangling header on blanks

    def test_nothing_renders_without_a_scope(self):
        assert render_memory_section({}) == ""
        # an empty scope still gets the whitelist — that is how the agent
        # learns it may write there — just no heading
        rendered = render_memory_section({MemoryScope.TEAM: "   "})
        assert "- `team` (" in rendered
        assert "### Team Memory" not in rendered


class TestParseSynthesis:
    def test_tolerates_a_fence_and_prose_around_the_object(self):
        out = parse_synthesis(
            'Here you go:\n```json\n{"constraints": [{"date": "2026-01-01", "text": "x"}], '
            '"summary": "s", "open_questions": []}\n```'
        )
        assert out.summary == "s" and out.constraints[0].text == "x"

    def test_prose_with_braces_of_its_own_does_not_fool_the_extraction(self):
        out = extract_json_object('Schema: {format: json}\n{"summary": "s"}\ntrailing')
        assert out == {"summary": "s"}
        with pytest.raises(ValueError):
            extract_json_object("no json here {not: json}")

    def test_a_malformed_shape_degrades_instead_of_failing_the_rebuild(self):
        """The model's slip must not leave the topic unsummarized until the
        sweep: an item without text is dropped, a non-string summary is
        blank, a date in the wrong shape is blank."""
        out = parse_synthesis(
            '{"constraints": ["bare text", {"date": "Jan 2026", "text": "kept"}, '
            '{"date": "x"}], "summary": ["not", "a", "string"], '
            '"open_questions": ["", "asked?", 3]}'
        )
        assert [(c.date, c.text) for c in out.constraints] == [("", "kept")]
        assert out.summary == ""
        assert out.open_questions == ["asked?"]
        assert MemoryConstraint(date="2026-09-06T10:00", text="t").date == "2026-09-06"


# --- account page -------------------------------------------------------------


class TestAccountMemories:
    @pytest.fixture(autouse=True)
    def _no_agent_info_lookups(self, monkeypatch):
        """Agent-info enrichment needs Redis and the lead-name fallback needs
        the teams table; stub both out."""

        async def fake_get_agent_infos(agent_ids):
            return {}

        async def fake_lead_config(team_id):
            return None

        monkeypatch.setattr(
            "intentkit.core.agent.info.get_agent_infos", fake_get_agent_infos
        )
        monkeypatch.setattr(
            "intentkit.models.team.Team.get_lead_agent_config", fake_lead_config
        )

    @pytest.mark.asyncio
    async def test_lists_only_own_team_and_user_rows(self, memory_tables):
        synth = FakeSynthesizer()
        await record_memory(
            team_ref(), claim="team doc", user_stated=True, synthesizer=synth
        )
        await record_memory(
            team_ref("domain_terms", agent_id="team-team-1"),
            claim="lead doc",
            user_stated=True,
            synthesizer=synth,
        )
        await record_memory(
            TopicRef("agent-1", MemoryScope.USER, "user-1", "user_preferences"),
            claim="user doc",
            user_stated=True,
            synthesizer=synth,
        )
        # None of these belong to (team-1, user-1):
        await record_memory(
            TopicRef("agent-1", MemoryScope.TEAM, "team-2", "team_profile"),
            claim="other team",
            user_stated=True,
            synthesizer=synth,
        )
        await record_memory(
            TopicRef("agent-1", MemoryScope.USER, "user-2", "user_preferences"),
            claim="other user",
            user_stated=True,
            synthesizer=synth,
        )
        await record_memory(
            TopicRef("agent-1", MemoryScope.CRON, "task-1", "task_setup"),
            claim="cron doc",
            user_stated=True,
            synthesizer=synth,
        )
        await record_memory(
            TopicRef("agent-1", MemoryScope.CHANNEL, "chat-1", "channel_purpose"),
            claim="channel doc",
            user_stated=True,
            synthesizer=synth,
        )
        # a topic whose rebuild died has no row on the page (only the prompt's fallback)
        await record_memory(
            team_ref("hard_rules"),
            claim="unsummarized",
            user_stated=True,
            synthesizer=None,
        )

        memories = await list_account_memories("team-1", "user-1")

        assert [(m.scope, m.agent_id, m.topic) for m in memories] == [
            ("team", "agent-1", "team_profile"),
            ("team", "team-team-1", "domain_terms"),
            ("user", "agent-1", "user_preferences"),
        ]
        assert memories[0].topic_label == "Team profile"
        assert memories[0].summary == "synthesized"

    @pytest.mark.asyncio
    async def test_labels_lead_agent(self, memory_tables):
        await record_memory(
            team_ref(agent_id="team-team-1"),
            claim="lead doc",
            user_stated=True,
            synthesizer=FakeSynthesizer(),
        )
        with patch(
            "intentkit.models.team.Team.get_lead_agent_config",
            new=AsyncMock(return_value={"name": "Concierge", "avatar": "lead.png"}),
        ):
            memories = await list_account_memories("team-1", "user-1")
        assert memories[0].agent_name == "Concierge"
        assert memories[0].agent_picture == "lead.png"

        with patch(
            "intentkit.models.team.Team.get_lead_agent_config",
            new=AsyncMock(return_value=None),
        ):
            memories = await list_account_memories("team-1", "user-1")
        assert memories[0].agent_name == "Team Lead"

    @pytest.mark.asyncio
    async def test_a_cleared_topic_is_left_off_the_page(self, memory_tables):
        ref = team_ref("hard_rules")
        await record_memory(
            ref, claim="a finding", user_stated=False, synthesizer=FakeSynthesizer()
        )
        assert len(await list_account_memories("team-1", "user-1")) == 1

        await sweep_memory(datetime.now(UTC) + timedelta(days=91), FakeSynthesizer())
        assert await list_account_memories("team-1", "user-1") == []


# --- tool binding -------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in for ModelRequest: runtime.context plus override()."""

    def __init__(self, context: AgentContext) -> None:
        self.runtime = SimpleNamespace(context=context)
        self.overridden: dict[str, Any] = {}

    def override(self, **kwargs: Any) -> "_FakeRequest":
        self.overridden.update(kwargs)
        return self


class TestRecordMemoryToolGating:
    """ToolBindingMiddleware binds record_memory only when a scope resolves."""

    @staticmethod
    async def _bound_tool_names(context: AgentContext) -> set[str]:
        from intentkit.core.middleware import ToolBindingMiddleware
        from intentkit.core.system_tools import current_time, record_memory

        llm_model = MagicMock()
        llm_model.create_instance = AsyncMock(return_value=MagicMock())
        middleware = ToolBindingMiddleware(llm_model, [current_time, record_memory])
        request = _FakeRequest(context)
        handler = AsyncMock(return_value="response")
        await middleware.awrap_model_call(cast(Any, request), handler)
        return {t.name for t in request.overridden["tools"]}

    @staticmethod
    def _agent_context(**overrides) -> AgentContext:
        agent = MagicMock()
        agent.team_id = overrides.pop("agent_team_id", "team-owner")
        defaults: dict[str, Any] = {
            "agent_id": "agent-1",
            "get_agent": lambda: agent,
            "chat_id": "chat-1",
            "user_id": "user-1",
            "team_id": "team-1",
            "entrypoint": AuthorType.WEB,
            "is_own_team": True,
        }
        defaults.update(overrides)
        return AgentContext(**defaults)

    @pytest.mark.asyncio
    async def test_bound_when_scopes_resolve(self):
        names = await self._bound_tool_names(self._agent_context())
        assert "record_memory" in names

    @pytest.mark.asyncio
    async def test_dropped_for_subagent_runs(self):
        names = await self._bound_tool_names(self._agent_context(call_depth=1))
        assert "record_memory" not in names
        assert "current_time" in names

    @pytest.mark.asyncio
    async def test_dropped_for_teamless_anonymous_guests(self):
        context = self._agent_context(user_id=None, team_id=None, is_own_team=False)
        names = await self._bound_tool_names(context)
        assert "record_memory" not in names
        assert "current_time" in names
