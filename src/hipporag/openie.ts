/**
 * openie.ts — Combined summarization, triple extraction, and tagging.
 *
 * A single Haiku call processes each exchange into:
 *   1. A dense summary (4:1 compression)
 *   2. Knowledge triples (subject, predicate, object)
 *   3. Two-level tags (type + topics) for metadata filtering
 *   4. A salience score
 *
 * This replaces the separate summarization and extraction steps —
 * one LLM call does everything, keeping API costs low.
 */

import type { Triple, PassageTags, NoteType } from "./types.ts";
import { llmFast } from "../llm.ts";
import { extractJsonFromResponse, normalizeEntity } from "../utils.ts";

/** Result of processing a conversation exchange */
export interface ProcessResult {
  /** Dense summary of the exchange (~25% of original) */
  summary: string;
  /** Extracted (subject, predicate, object) triples */
  triples: Triple[];
  /** Two-level tags for filtering */
  tags: PassageTags;
  /** Importance score 0-1 */
  salience: number;
}

const VALID_TYPES: NoteType[] = [
  "event",
  "decision",
  "preference",
  "fact",
  "goal",
  "plan",
];

/**
 * Process a conversation exchange: summarize + extract triples + tag.
 *
 * Single LLM call that produces everything needed for HippoRAG indexing.
 *
 * @param text — the conversation exchange text
 * @returns { summary, triples, tags, salience }
 */
export async function processExchange(
  text: string
): Promise<ProcessResult> {
  try {
    const response = await llmFast.invoke([
      {
        role: "system" as const,
        content: `You process conversation exchanges into structured memory. Return ONLY valid JSON.

{
  "summary": "Dense 1-3 sentence summary of the exchange. Include key facts, dates, numbers, decisions. Skip pleasantries and generic advice.",
  "triples": [
    {"subject": "entity1", "predicate": "relation", "object": "entity2"}
  ],
  "tags": {
    "type": ["event", "decision"],
    "topics": ["property", "cedar-creek"]
  },
  "salience": 0.7
}

Summary rules:
- Compress to ~25% of original length
- Focus on facts established: decisions, events, numbers, dates, preferences
- Include both user and assistant facts that matter (prices, names, specifics)
- Skip generic advice the assistant could regenerate

Triple rules:
- Extract concrete entities and relationships
- Normalize entity names to lowercase (1-3 words)
- Use simple predicates: works_at, likes, viewed, purchased, costs, located_in, decided_on
- 1-5 triples per exchange — only the most important facts

Tag rules:
- type (pick from FIXED list): event, decision, preference, fact, goal, plan
  - event: something that happened (viewed property, bought item, had meeting)
  - decision: a choice was made (chose quartz, went with AHS)
  - preference: a like/dislike/want (wants pool, hates noise)
  - fact: established data (price $340k, 4% interest, 10-mile commute)
  - goal: something to achieve (1000 miles by summer)
  - plan: future intention (schedule walk-through, order rack next week)
- topics (OPEN list): lowercase keywords for what it's about
  - Use specific terms: "property", "brookside", "kitchen", "mortgage", "cycling"
  - Include entity names that might be searched: "cedar-creek", "ahs", "quartz"
  - 2-5 topic tags per exchange

Salience (0-1):
- 0.1-0.3: small talk, greetings
- 0.4-0.6: routine discussion
- 0.7-0.9: important facts, decisions, events
- 1.0: critical commitments`,
      },
      {
        role: "user" as const,
        content: text,
      },
    ]);

    const responseText =
      typeof response.content === "string"
        ? response.content
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("");

    const jsonStr = extractJsonFromResponse(responseText);
    const parsed = JSON.parse(jsonStr) as {
      summary: string;
      triples: Array<{ subject: string; predicate: string; object: string }>;
      tags: { type: string[]; topics: string[] };
      salience: number;
    };

    // Normalize triples
    const triples: Triple[] = (parsed.triples ?? []).map((t) => ({
      subject: normalizeEntity(t.subject),
      predicate: t.predicate.toLowerCase().trim(),
      object: normalizeEntity(t.object),
    }));

    // Validate and normalize tags
    const validTypes = (parsed.tags?.type ?? []).filter((t): t is NoteType =>
      VALID_TYPES.includes(t as NoteType)
    );
    const topics = (parsed.tags?.topics ?? []).map((t) =>
      t.toLowerCase().trim()
    );

    const tags: PassageTags = {
      type: validTypes.length > 0 ? validTypes : ["fact"],
      topics: topics.length > 0 ? topics : [],
    };

    const salience = Math.max(0, Math.min(1, parsed.salience ?? 0.5));
    const summary = parsed.summary?.trim() ?? text.slice(0, 200);

    return { summary, triples, tags, salience };
  } catch (error) {
    console.error("[OpenIE] Failed to process exchange:", error);
    return {
      summary: text.slice(0, 200),
      triples: [],
      tags: { type: ["fact"], topics: [] },
      salience: 0.5,
    };
  }
}

/**
 * Convert a triple to a string for embedding.
 */
export function tripleToString(triple: Triple): string {
  return `${triple.subject} ${triple.predicate} ${triple.object}`;
}
