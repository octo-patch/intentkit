"use client";

import { useQuery } from "@tanstack/react-query";

import { MemorySections } from "@/components/features/MemorySections";
import { memoryApi } from "@/lib/api";

export default function ClientPage() {
  const {
    data: memories = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["memories"],
    queryFn: () => memoryApi.list(),
    staleTime: 30_000,
  });

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold">Memory</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Agents keep these memories themselves as they work: what people told
          them, what they worked out, and what they still mean to confirm.
          To change something, tell the agent in a conversation — it records
          the correction as a newer entry.
        </p>
      </div>
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading memories...</p>
      ) : error ? (
        <p className="text-sm text-destructive">Failed to load memories.</p>
      ) : (
        <MemorySections memories={memories} />
      )}
    </div>
  );
}
