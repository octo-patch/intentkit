"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { MemorySummary } from "@/types/memory";

function agentTitle(memory: MemorySummary): string {
  return memory.agent_name || memory.agent_id;
}

/** Group the flat summary list by agent, keeping the server's order. */
function groupByAgent(memories: MemorySummary[]): MemorySummary[][] {
  const groups = new Map<string, MemorySummary[]>();
  for (const memory of memories) {
    const group = groups.get(memory.agent_id);
    if (group) {
      group.push(memory);
    } else {
      groups.set(memory.agent_id, [memory]);
    }
  }
  return Array.from(groups.values());
}

function TopicBlock({ memory }: { memory: MemorySummary }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold">{memory.topic_label}</h3>
      {memory.constraints.length > 0 && (
        <div>
          <p className="text-xs font-medium text-muted-foreground">
            Told by people
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm">
            {memory.constraints.map((constraint, index) => (
              <li key={index}>
                {constraint.date && (
                  <span className="mr-1.5 font-mono text-xs text-muted-foreground">
                    {constraint.date}
                  </span>
                )}
                {constraint.text}
              </li>
            ))}
          </ul>
        </div>
      )}
      {memory.summary && (
        <div>
          <p className="text-xs font-medium text-muted-foreground">
            Worked out by the agent
          </p>
          <p className="mt-1 whitespace-pre-wrap text-sm">{memory.summary}</p>
        </div>
      )}
      {memory.open_questions.length > 0 && (
        <div>
          <p className="text-xs font-medium text-muted-foreground">
            To confirm
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm">
            {memory.open_questions.map((question, index) => (
              <li key={index}>{question}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function AgentCard({ memories }: { memories: MemorySummary[] }) {
  const latest = memories.reduce((a, b) =>
    a.synthesized_at > b.synthesized_at ? a : b,
  );
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base truncate" title={agentTitle(latest)}>
          {agentTitle(latest)}
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          Updated {new Date(latest.synthesized_at).toLocaleString()}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {memories.map((memory) => (
          <TopicBlock key={memory.id} memory={memory} />
        ))}
      </CardContent>
    </Card>
  );
}

function MemoryGroup({
  title,
  description,
  memories,
  emptyText,
}: {
  title: string;
  description: string;
  memories: MemorySummary[];
  emptyText: string;
}) {
  const agents = groupByAgent(memories);
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {agents.length === 0 ? (
        <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
          {emptyText}
        </p>
      ) : (
        <div className="space-y-3">
          {agents.map((group) => (
            <AgentCard key={group[0].agent_id} memories={group} />
          ))}
        </div>
      )}
    </section>
  );
}

export function MemorySections({ memories }: { memories: MemorySummary[] }) {
  const teamMemories = memories.filter((m) => m.scope === "team");
  const userMemories = memories.filter((m) => m.scope === "user");

  return (
    <div className="space-y-8">
      <MemoryGroup
        title="Team Memory"
        description="What each agent remembers for the whole team, by topic."
        memories={teamMemories}
        emptyText="No team memory yet. Agents record it themselves as your team works with them."
      />
      <MemoryGroup
        title="Your Memory"
        description="What each agent remembers about you personally, by topic."
        memories={userMemories}
        emptyText="No personal memory yet. Agents record it themselves as you chat with them."
      />
    </div>
  );
}
