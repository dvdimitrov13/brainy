/**
 * openie.ts — Combined summarization, triple extraction, and tagging.
 *
 * A single Haiku call processes each exchange using tool-based structured
 * output — the LLM "calls" a tool whose schema defines the exact output
 * format, guaranteeing valid JSON.
 */

import type { Triple, PassageTags, NoteType } from "./types.ts";
import { llmFast } from "../llm.ts";
import { normalizeEntity } from "../utils.ts";

/** Result of processing a conversation exchange */
export interface ProcessResult {
  summary: string;
  triples: Triple[];
  tags: PassageTags;
  salience: number;
}

const VALID_TYPES: NoteType[] = [
  "event", "decision", "preference", "fact", "goal", "plan",
];

/** Tool schema for structured output */
const PROCESS_TOOL = {
  type: "function" as const,
  function: {
    name: "process_exchange",
    description: "Process a conversation exchange into structured memory.",
    parameters: {
      type: "object" as const,
      properties: {
        summary: {
          type: "string" as const,
          description:
            "Dense 1-3 sentence summary. Include key facts, dates, numbers, decisions. Skip pleasantries and generic advice. Compress to ~25% of original.",
        },
        triples: {
          type: "array" as const,
          items: {
            type: "object" as const,
            properties: {
              subject: { type: "string" as const, description: "Entity name, lowercase, 1-3 words" },
              predicate: { type: "string" as const, description: "Relationship: works_at, likes, viewed, purchased, costs, decided_on, etc." },
              object: { type: "string" as const, description: "Entity name, lowercase, 1-3 words" },
            },
            required: ["subject", "predicate", "object"],
          },
          description: "1-5 knowledge triples — only the most important facts.",
        },
        tags: {
          type: "object" as const,
          properties: {
            type: {
              type: "array" as const,
              items: {
                type: "string" as const,
                enum: ["event", "decision", "preference", "fact", "goal", "plan"],
              },
              description: "What kind of information: event (happened), decision (choice made), preference (like/dislike), fact (data), goal (target), plan (intention).",
            },
            topics: {
              type: "array" as const,
              items: { type: "string" as const },
              description: "2-5 lowercase topic keywords: property, brookside, kitchen, cycling. Include searchable entity names.",
            },
          },
          required: ["type", "topics"],
        },
        salience: {
          type: "number" as const,
          description: "Importance 0-1. 0.1-0.3: small talk. 0.4-0.6: routine. 0.7-0.9: important facts/decisions. 1.0: critical.",
        },
      },
      required: ["summary", "triples", "tags", "salience"],
    },
  },
};

/**
 * Process a conversation exchange: summarize + extract triples + tag.
 * Uses tool-based structured output for guaranteed valid JSON.
 */
export async function processExchange(
  text: string
): Promise<ProcessResult> {
  try {
    const response = await llmFast.invoke(
      [
        {
          role: "system" as const,
          content: "Process the conversation exchange into structured memory by calling the process_exchange tool.",
        },
        { role: "user" as const, content: text },
      ],
      {
        tools: [PROCESS_TOOL],
        tool_choice: { type: "tool" as const, name: "process_exchange" },
      }
    );

    // Extract tool call args — guaranteed valid JSON by the API
    const toolCall = response.tool_calls?.[0];
    if (!toolCall) throw new Error("No tool call returned");

    const args = toolCall.args as {
      summary: string;
      triples: Array<{ subject: string; predicate: string; object: string }>;
      tags: { type: string[]; topics: string[] };
      salience: number;
    };

    const triples: Triple[] = (args.triples ?? []).map((t) => ({
      subject: normalizeEntity(t.subject),
      predicate: t.predicate.toLowerCase().trim(),
      object: normalizeEntity(t.object),
    }));

    const validTypes = (args.tags?.type ?? []).filter((t): t is NoteType =>
      VALID_TYPES.includes(t as NoteType)
    );

    const tags: PassageTags = {
      type: validTypes.length > 0 ? validTypes : ["fact"],
      topics: (args.tags?.topics ?? []).map((t) => t.toLowerCase().trim()),
    };

    return {
      summary: args.summary?.trim() ?? text.slice(0, 200),
      triples,
      tags,
      salience: Math.max(0, Math.min(1, args.salience ?? 0.5)),
    };
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

/** Convert a triple to a string for embedding. */
export function tripleToString(triple: Triple): string {
  return `${triple.subject} ${triple.predicate} ${triple.object}`;
}
