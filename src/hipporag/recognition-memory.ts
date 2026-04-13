/**
 * recognition-memory.ts — LLM-based triple filtering ("Recognition Memory").
 *
 * Uses tool-based structured output for guaranteed valid JSON.
 * The LLM "calls" a filter_triples tool with the indices of relevant triples.
 */

import type { Triple } from "./types.ts";
import { llmMid } from "../llm.ts";

/** Tool schema for structured output */
const FILTER_TOOL = {
  type: "function" as const,
  function: {
    name: "filter_triples",
    description: "Select which triples are relevant to the query.",
    parameters: {
      type: "object" as const,
      properties: {
        relevant_indices: {
          type: "array" as const,
          items: { type: "number" as const },
          description: "1-based indices of the relevant triples. Empty array if none are relevant.",
        },
      },
      required: ["relevant_indices"],
    },
  },
};

/**
 * Filter candidate triples to keep only those relevant to the query.
 * Uses tool-based structured output for reliable parsing.
 */
export async function filterTriples(
  query: string,
  candidateTriples: Triple[]
): Promise<Triple[]> {
  if (candidateTriples.length === 0) return [];
  if (candidateTriples.length <= 2) return candidateTriples;

  try {
    const triplesFormatted = candidateTriples
      .map(
        (t, i) =>
          `${i + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
      )
      .join("\n");

    const response = await llmMid.invoke(
      [
        {
          role: "system" as const,
          content: `You are a relevance filter. Given a query and knowledge triples, call filter_triples with the indices of triples that are actually relevant.

A triple is relevant if it directly relates to the query, provides needed context, or connects important entities.
A triple is NOT relevant if it mentions unrelated entities or irrelevant relationships.`,
        },
        {
          role: "user" as const,
          content: `Query: ${query}\n\nCandidate triples:\n${triplesFormatted}`,
        },
      ],
      {
        tools: [FILTER_TOOL],
        tool_choice: { type: "tool" as const, name: "filter_triples" },
      }
    );

    const toolCall = response.tool_calls?.[0];
    if (!toolCall) return candidateTriples;

    const args = toolCall.args as { relevant_indices: number[] };

    return (args.relevant_indices ?? [])
      .map((i) => candidateTriples[i - 1])
      .filter((t): t is Triple => t !== undefined);
  } catch (error) {
    console.error("[RecognitionMemory] Failed to filter triples:", error);
    return candidateTriples;
  }
}
