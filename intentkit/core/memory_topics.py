"""Memory topics — the closed whitelist, per scope.

Memory is filed by **scope** (whose memory it is) and **topic** (what may be
kept). The whitelist is the guard: a fact that fits no topic is not recorded
at all. Without one, a free-form memory fills with the last conversation
instead of what the team taught — which is what the single merged note per
scope actually did.

Topics are routing keys the system prompt and the tool both name, so they
live in code and version with it. A topic earns its slot only if some run
READS it: the list is derived from what an agent does for a team — answer
people, research, produce content, drive tools, fire a scheduled task —
never invented on the write side.

``ttl_days`` bounds **findings only**. What a person stated never expires:
silently dropping an instruction someone gave is worse than letting it age.
The number answers one question — how long is something the agent noticed
by itself still worth believing unrefreshed.
"""

from dataclasses import dataclass

from intentkit.models.memory import MemoryScope


@dataclass(frozen=True, slots=True)
class TopicSpec:
    """One registered topic.

    ``guidance`` is write-side copy — what belongs here, and what does not.
    It is rendered into the system prompt (where the agent decides WHETHER
    to record, before any tool call) as well as into the tool's failure
    messages. A tool schema alone would arrive too late for that decision.
    """

    value: str
    label: str
    guidance: str
    ttl_days: int


# A topic dropped from this registry keeps its rows; its findings age out on
# this instead of lingering forever under a name nothing reads any more.
RETIRED_TOPIC_TTL_DAYS = 180


SCOPE_TOPICS: dict[MemoryScope, tuple[TopicSpec, ...]] = {
    MemoryScope.TEAM: (
        TopicSpec(
            "team_profile",
            "Team profile",
            "What this team is and does as it bears on your work for them: "
            "the business, the products or lines of work, which one is the "
            "main one, and a line on what each is. An orientation, not a "
            "dossier — the particulars of the work go under `subject_facts`.",
            365,
        ),
        TopicSpec(
            "subject_facts",
            "Subject facts",
            "The durable facts about the work you do for this team: the "
            "market or the rivals you track, the clients or partners "
            "involved, the accounts you publish to, what has been decided "
            "about that work. Facts about the subject, never status or "
            "progress, and never how you should behave — that is "
            "`working_agreements`.",
            180,
        ),
        TopicSpec(
            "people_and_roles",
            "People and roles",
            "Who is who and who owns what — the mapping that decides whom to "
            "name, route to, or wait on. Durable ownership only, not who "
            "happens to be handling something this week.",
            365,
        ),
        TopicSpec(
            "working_agreements",
            "Working agreements",
            "How this team wants you to work: when to act versus ask first, "
            "when to stay out of it, how much effort to spend on a question, "
            "escalation. Standing rules for the whole team, not one person's "
            "preferences — those go to that person's `user_preferences`.",
            180,
        ),
        TopicSpec(
            "output_conventions",
            "Output conventions",
            "Recurring expectations about the shape of the answer: language, "
            "length, format, how recurring reports are laid out, where results "
            "get delivered. Not one-off requests.",
            180,
        ),
        TopicSpec(
            "team_resources",
            "Resources",
            "The canonical place for a kind of work — repo, shared folder, "
            "board, doc, tracker, account — recorded with its NAME, its LINK "
            "or id, and WHICH work it serves. Never credentials, never an "
            "inventory of everything.",
            365,
        ),
        TopicSpec(
            "domain_terms",
            "Domain terms",
            "The team's own vocabulary: internal names, acronyms, how a metric "
            "or a status is defined here. What a newcomer would have to be "
            "told before the rest of the memory reads correctly.",
            365,
        ),
        TopicSpec(
            "hard_rules",
            "Hard rules",
            "Red lines: what you must never do, say, or publish for this team, "
            "and the constraints behind them. One rejection is not a standing "
            "rule — record the rule when it is stated as one.",
            90,
        ),
    ),
    MemoryScope.USER: (
        TopicSpec(
            "user_preferences",
            "Preferences",
            "How this person wants their own answers: language, format, "
            "detail, timezone, how much to check in. A recurring preference, "
            "not this conversation's instruction.",
            180,
        ),
        TopicSpec(
            "user_context",
            "Role and focus",
            "Their role and what they durably own or work on, when it changes "
            "how to help them. Not their task list and not project status.",
            365,
        ),
        TopicSpec(
            "user_resources",
            "Personal resources",
            "Their own default repo, folder, account or workspace, with name "
            "and link — what to reach for when they name none.",
            365,
        ),
    ),
    MemoryScope.CHANNEL: (
        TopicSpec(
            "channel_purpose",
            "Purpose",
            "What this chat is for, who is in it, and which work is in scope "
            "here. Only what is specifically true of this chat.",
            365,
        ),
        TopicSpec(
            "channel_subject",
            "Subject",
            "The durable facts about what this chat is ABOUT: the product, "
            "client, project or market it concerns, and what has been decided "
            "about that work here. What is specific to this chat lives here; "
            "what holds across every chat goes to the team's `subject_facts`. "
            "Not how you should behave here, and not status or progress.",
            180,
        ),
        TopicSpec(
            "participation_rules",
            "Participation rules",
            "When you may answer, post, or must stay quiet in this chat, and "
            "whose instructions count here. Never infer permission from having "
            "posted before.",
            180,
        ),
        TopicSpec(
            "channel_conventions",
            "Conventions and defaults",
            "This chat's own defaults and recurring shapes: the language, the "
            "format of a recurring post, where results get delivered, local "
            "naming. Not a copy of the team's conventions, which you already "
            "read.",
            180,
        ),
    ),
    # An autonomous task's memory is its ONLY carry-over between runs — every
    # run starts with a fresh conversation by design. Inclusion test: will
    # this change what a future run does? A run with nothing to carry forward
    # records nothing; there is no run diary. Caps are generous because a
    # missed run must not erase the state that a late one needs.
    MemoryScope.CRON: (
        TopicSpec(
            "task_setup",
            "Setup and scope",
            "The resolved context every run needs: ids, destinations, "
            "inclusion rules, who to address. Never a copy of the task's own "
            "prompt or schedule; a fact true beyond this task goes to the team.",
            365,
        ),
        TopicSpec(
            "source_cursors",
            "Cursors",
            "Where the last run got to — the timestamp, id, or boundary that "
            "lets the next one resume without gaps or repeats. The newest "
            "cursor is the whole state, never a history of them.",
            365,
        ),
        TopicSpec(
            "dedup_state",
            "Already covered",
            "A bounded rolling list of what has already been posted or "
            "handled, within this task's no-repeat horizon. Never the content "
            "itself.",
            180,
        ),
        TopicSpec(
            "baselines",
            "Baselines",
            "The snapshot or running tally the next run compares against — "
            "only the fields it actually compares. The newest replaces the "
            "last; never a narrative of past runs.",
            180,
        ),
        TopicSpec(
            "open_threads",
            "Blockers and next steps",
            "What is unresolved and what the next run should do about it — "
            "the blocker with its conditional action, the question still "
            "waiting on a person. Record the resolution to supersede it.",
            180,
        ),
        TopicSpec(
            "method_notes",
            "Method notes",
            "What this task learned the hard way and must not rediscover: a "
            "quirk of a source, a query that works, a trap. A lesson true "
            "beyond this task goes to the team scope.",
            365,
        ),
    ),
}

# The order every surface renders scopes in: broadest first, so what the
# whole team taught frames what one chat, one person or one task did.
# ``resolve_memory_scopes`` builds a run's scopes from this and the prompt
# walks it, so there is one ordering, not two that agree by coincidence.
SCOPE_ORDER: tuple[MemoryScope, ...] = (
    MemoryScope.TEAM,
    MemoryScope.CHANNEL,
    MemoryScope.USER,
    MemoryScope.CRON,
)

SCOPE_TITLES: dict[MemoryScope, str] = {
    MemoryScope.TEAM: "Team Memory",
    MemoryScope.USER: "User Memory",
    MemoryScope.CHANNEL: "Channel Memory",
    MemoryScope.CRON: "Cron Task Memory",
}

SCOPE_BLURBS: dict[MemoryScope, str] = {
    MemoryScope.TEAM: "shared by everyone on the team you are working for",
    MemoryScope.USER: "this person, across every conversation they have with you",
    MemoryScope.CHANNEL: "this chat on a channel platform, shared by everyone in it",
    MemoryScope.CRON: "this scheduled task — its only carry-over between runs",
}

# Render order for topics inside a block, so a scope's memory reads the same
# way every turn no matter what order the summaries came back in.
TOPIC_ORDER: dict[tuple[MemoryScope, str], int] = {
    (scope, spec.value): i
    for scope, specs in SCOPE_TOPICS.items()
    for i, spec in enumerate(specs)
}


def topic_spec(scope: MemoryScope, value: str) -> TopicSpec | None:
    return next((s for s in SCOPE_TOPICS[scope] if s.value == value), None)


def topic_label(scope: MemoryScope, value: str) -> str:
    spec = topic_spec(scope, value)
    return spec.label if spec else value


def topic_names(scope: MemoryScope) -> str:
    """The scope's topics as one comma-joined line — for a failure message
    that has to name what WAS allowed."""
    return ", ".join(f"`{s.value}`" for s in SCOPE_TOPICS[scope])


def topic_guidance(scope: MemoryScope) -> str:
    """The scope's topics with what belongs in each, one per line.

    Every model that has to choose a topic reads this — the agent through
    the prompt's whitelist, the legacy-note importer through its own
    instructions — so they cannot be shown different versions of the list.
    """
    return "\n".join(f"- `{s.value}` — {s.guidance}" for s in SCOPE_TOPICS[scope])


def topic_rank(scope: MemoryScope, topic: str) -> tuple[int, str]:
    """Sort key: registry order, and a topic no longer registered sorts
    last. Used by the prompt block and by the account page, which must
    agree on what "last" means."""
    return (TOPIC_ORDER.get((scope, topic), len(TOPIC_ORDER)), topic)
