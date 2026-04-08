/**
 * recognition-memory.ts — LLM-based triple filtering ("Recognition Memory").
 *
 * This is HippoRAG2's key improvement over HippoRAG1. In the brain,
 * "recognition memory" is the ability to distinguish things you've
 * actually encountered before from things that just seem familiar.
 *
 * In retrieval terms: dense embedding search finds triples that are
 * semantically SIMILAR to the query, but not all of them are actually
 * RELEVANT. This module uses the LLM as a "recognition filter" to
 * separate truly relevant triples from false positives.
 *
 * Pipeline position:
 *   1. Dense retrieval finds top-K candidate triples by embedding similarity
 *   2. ➡️ THIS MODULE filters them down to only the truly relevant ones
 *   3. The surviving triples seed the PPR graph search
 *
 * Without this step (HippoRAG1), noisy triples pollute the PPR seeds
 * and degrade multi-hop reasoning quality.
 *
 * TS note for Python devs:
 *   - `Array<T>` and `T[]` are the same thing — both mean "array of T".
 *     We use `Array<T>` when the inner type is complex for readability.
 */

import type { Triple } from "./types.ts";
import { llm } from "../llm.ts";
import { extractJsonFromResponse } from "../utils.ts";

/**
 * Filter candidate triples to keep only those relevant to the query.
 *
 * This asks the LLM: "Given this query, which of these triples are
 * actually relevant for answering it?"
 *
 * In the HippoRAG2 paper, this uses DSPy MIPROv2 for prompt optimisation.
 * We use a simpler direct prompt since we're not doing automated
 * prompt tuning — the results are still effective.
 *
 * @param query           — the user's query/question
 * @param candidateTriples — triples retrieved by dense search
 * @returns the filtered subset of triples that are relevant
 */
export async function filterTriples(
  query: string,
  candidateTriples: Triple[]
): Promise<Triple[]> {
  // If there are no candidates, nothing to filter
  if (candidateTriples.length === 0) return [];

  // If only 1-2 candidates, don't bother filtering (save an LLM call)
  if (candidateTriples.length <= 2) return candidateTriples;

  try {
    // Format triples as a numbered list for the LLM
    const triplesFormatted = candidateTriples
      .map(
        (t, i) =>
          `${i + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
      )
      .join("\n");

    const response = await llm.invoke([
      {
        role: "system" as const,
        content: `You are a relevance filter. Given a query and a list of knowledge triples, identify which triples are actually relevant for answering the query.

A triple is relevant if:
- It directly relates to what the query is asking about
- It provides context or background needed to answer the query
- It connects entities that are important for the query

A triple is NOT relevant if:
- It mentions entities unrelated to the query
- It describes relationships irrelevant to what's being asked
- It's topically similar but doesn't help answer the query

Return ONLY valid JSON with this format:
{
  "relevant_indices": [1, 3, 5]
}

Where the numbers are the 1-based indices of the relevant triples.
If NONE are relevant, return: {"relevant_indices": []}`,
      },
      {
        role: "user" as const,
        content: `Query: ${query}

Candidate triples:
${triplesFormatted}

Which triples are relevant to answering this query?`,
      },
    ]);

    // Parse the LLM response
    const responseText =
      typeof response.content === "string"
        ? response.content
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("");

    const jsonStr = extractJsonFromResponse(responseText);
    const parsed = JSON.parse(jsonStr) as { relevant_indices: number[] };

    // Convert 1-based indices back to triples
    // `filter(Boolean)` removes any undefined values (if index was out of range)
    // Python equivalent: [candidates[i-1] for i in indices if 0 < i <= len(candidates)]
    const filtered = parsed.relevant_indices
      .map((i) => candidateTriples[i - 1])
      .filter((t): t is Triple => t !== undefined);

    return filtered;
  } catch (error) {
    // On failure, return all candidates unfiltered (graceful degradation)
    console.error("[RecognitionMemory] Failed to filter triples:", error);
    return candidateTriples;
  }
}
