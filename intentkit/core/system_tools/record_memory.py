"""Tool for recording one durable fact into the agent's scoped memory."""

from typing import Annotated, override

from langchain_core.tools import ArgsSchema, InjectedToolCallId
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from intentkit.core.system_tools.base import SystemTool
from intentkit.models.memory import MemoryScope


class RecordMemoryInput(BaseModel):
    """Input schema for recording a memory entry."""

    scope: MemoryScope = Field(
        ...,
        description=(
            "Whose memory this belongs to: 'team' for what holds for the "
            "whole team, or this conversation's own scope ('user', 'channel' "
            "or 'cron') for what is about this person, this chat or this "
            "scheduled task. Only the scopes listed in your Memory section "
            "are available."
        ),
    )
    topic: str = Field(
        ...,
        description=(
            "The topic to file the fact under, exactly as named in the "
            "Memory section's whitelist for that scope."
        ),
    )
    claim: str = Field(
        ...,
        description=(
            "The fact as one self-contained sentence. It must carry any "
            "reference a later run has to follow — a link, an id, an exact "
            "name — because the claim is what you read back and the "
            "evidence is not."
        ),
    )
    user_stated: bool = Field(
        ...,
        description=(
            "True when a person told you this; false when you worked it out "
            "yourself. What a person said outranks what you found, "
            "permanently, and never expires."
        ),
    )
    evidence: str = Field(
        default="",
        description=(
            "Why you believe it: the person's own words, a source, a "
            "number. Kept for the record, not read back into your prompt."
        ),
    )


class RecordMemoryTool(SystemTool):
    """Tool that appends one entry to a scoped memory and rebuilds its topic.

    Entries are append-only: a correction is a newer entry, and the topic's
    summary — rebuilt with a cheap model call inside this tool — is where
    the newer one displaces the older. The tool checks only what the store
    cannot: that the scope is active in this conversation. The topic
    whitelist and the claim's shape are the store's to enforce, and its
    refusals are worded for the model to read back.
    """

    name: str = "record_memory"
    description: str = (
        "Keep one durable fact past this conversation. Pick the scope and "
        "the topic from the whitelist in your Memory section, state the "
        "fact as a single sentence in `claim`, say whether a person stated "
        "it, and put the supporting detail in `evidence`. Entries are "
        "append-only: correct a wrong one by recording the corrected "
        "statement. Sub-agents cannot record — when one's report turns up "
        "something worth keeping, this is the tool that keeps it."
    )
    args_schema: ArgsSchema | None = RecordMemoryInput
    requires_memory_scope: bool = True

    @override
    async def _arun(
        self,
        scope: MemoryScope,
        topic: str,
        claim: str,
        user_stated: bool,
        evidence: str = "",
        tool_call_id: Annotated[str | None, InjectedToolCallId] = None,
    ) -> str:
        from intentkit.core.memory import (
            MemoryInputError,
            TopicRef,
            record_memory,
            resolve_memory_scopes,
            try_create_memory_synthesizer,
        )

        context = self.get_context()
        if context.is_subagent:
            raise ToolException(
                "Memory is not available in sub-agent runs: the entry agent "
                "owns the conversation's memory."
            )
        available = {s.scope: s for s in resolve_memory_scopes(context.agent, context)}
        target = available.get(scope)
        if target is None:
            named = ", ".join(f"`{s.value}`" for s in available)
            raise ToolException(
                f"No `{scope.value}` memory in this conversation — available "
                f"here: {named}."
            )

        # The rebuild is billed like any internal tool LLM call.
        synthesizer = await try_create_memory_synthesizer(
            bill=lambda response, model_id: self._bill_internal_llm(
                response, tool_call_id, model_id
            ),
            metadata={"agent_id": context.agent_id, "chat_id": context.chat_id},
        )
        try:
            await record_memory(
                TopicRef(context.agent_id, scope, target.scope_key, topic),
                claim=claim,
                user_stated=user_stated,
                evidence=evidence,
                source_user=context.user_id or "",
                source_chat=context.chat_id,
                synthesizer=synthesizer,
            )
        except MemoryInputError as e:
            raise ToolException(str(e)) from e
        except Exception as e:
            self.logger.exception("record_memory failed")
            raise ToolException(f"Failed to record memory: {e}") from e

        kind = "as stated by them" if user_stated else "as your own finding"
        return (
            f"Recorded to {scope.value} memory under `{topic}` {kind}; it "
            "reads back next turn."
        )
