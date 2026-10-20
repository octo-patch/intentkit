/**
 * TypeScript types for Memory API responses
 */

export type MemoryScope = "team" | "user" | "channel" | "cron";

/** One thing a person told the agent, with the date it was said. */
export interface MemoryConstraint {
  date: string;
  text: string;
}

/**
 * One topic's synthesized memory of one agent — what the agent's prompt
 * renders. Rebuilt from append-only entries after every write; read-only
 * from the web.
 */
export interface MemorySummary {
  id: string;
  agent_id: string;
  scope: MemoryScope;
  scope_key: string;
  topic: string;
  topic_label: string;
  constraints: MemoryConstraint[];
  summary: string;
  open_questions: string[];
  synthesized_at: string;
  created_at: string;
  updated_at: string;
  agent_name?: string | null;
  agent_picture?: string | null;
}
