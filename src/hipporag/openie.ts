/**
 * openie.ts — Open Information Extraction via LLM.
 *
 * This module extracts (subject, predicate, object) triples from text
 * using Claude Sonnet. It mirrors HippoRAG's `openie_openai_gpt.py`.
 *
 * In HippoRAG, the LLM acts as the "neocortex" — understanding language
 * and extracting structured knowledge (entities + relationships) from
 * unstructured conversation text.
 *
 * The LLM also rates the "salience" of each passage — how important or
 * memorable it is. This is used later by the semantic forgetting system
 * to decide which memories to prune.
 *
 * TS note for Python devs:
 *   - `async function` is like `async def` in Python
 *   - `try { ... } catch (e) { ... }` is like `try: ... except Exception as e: ...`
 *   - We use `as string` (type assertion) to tell TS "I know this is a string"
 *     — similar to Python's `cast(str, value)` but more common in TS.
 */

import type { Triple } from "./types.ts";
import { llm } from "../llm.ts";
import { extractJsonFromResponse, normalizeEntity } from "../utils.ts";

/** Result of extracting triples from a piece of text */
export interface OpenIEResult {
  /** The extracted (subject, predicate, object) triples */
  triples: Triple[];
  /** LLM-judged salience/importance score, 0-1 */
  salience: number;
}

/**
 * Extract knowledge triples and salience from a text passage.
 *
 * This sends the text to Claude with a carefully crafted prompt that asks
 * it to identify entities and their relationships, then return structured
 * JSON. This is the "Open Information Extraction" (OpenIE) step — "open"
 * because it doesn't require a predefined schema of entity types.
 *
 * HippoRAG uses this to build its knowledge graph: each triple becomes
 * edges (subject ↔ object) in the graph, with the predicate as the
 * relationship label.
 *
 * @param text — the conversation text to extract from
 * @returns { triples, salience } — the extracted triples and importance score
 */
export async function extractTriples(text: string): Promise<OpenIEResult> {
  try {
    const response = await llm.invoke([
      {
        role: "system" as const,
        content: `You are a knowledge extraction system. Extract factual triples and rate salience from conversation text.

Return ONLY valid JSON with this exact format:
{
  "triples": [
    {"subject": "entity1", "predicate": "relation", "object": "entity2"}
  ],
  "salience": 0.7
}

Rules for triple extraction:
- Extract concrete entities (people, places, things, concepts) and their relationships
- Normalize entity names to lowercase, keep them concise (1-3 words)
- Use simple, descriptive predicates (e.g., "works_at", "likes", "is_a", "lives_in", "discussed", "wants_to", "has")
- Extract between 1-5 triples per passage — focus on the most important facts
- If the text is just pleasantries with no factual content, return an empty triples array

Rules for salience scoring (0-1):
- 0.1-0.3: small talk, greetings, acknowledgements
- 0.4-0.6: routine information exchange, general discussion
- 0.7-0.9: important facts, personal details, decisions, plans, preferences
- 1.0: critical information (commitments, key revelations)`,
      },
      {
        role: "user" as const,
        content: text,
      },
    ]);

    // Extract the text content from the LLM response
    const responseText =
      typeof response.content === "string"
        ? response.content
        : // response.content can be an array of content blocks (Anthropic format)
          // In that case, concatenate all text blocks
          (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("");

    // Parse the JSON from the response (handling markdown fences etc.)
    const jsonStr = extractJsonFromResponse(responseText);
    const parsed = JSON.parse(jsonStr) as {
      triples: Array<{ subject: string; predicate: string; object: string }>;
      salience: number;
    };

    // Normalise entity names to lowercase + trim
    const triples: Triple[] = parsed.triples.map((t) => ({
      subject: normalizeEntity(t.subject),
      predicate: t.predicate.toLowerCase().trim(),
      object: normalizeEntity(t.object),
    }));

    // Clamp salience to [0, 1]
    const salience = Math.max(0, Math.min(1, parsed.salience ?? 0.5));

    return { triples, salience };
  } catch (error) {
    // If anything goes wrong (JSON parse error, API error, etc.),
    // return empty triples with neutral salience rather than crashing.
    // The conversation can continue without this memory being indexed.
    console.error("[OpenIE] Failed to extract triples:", error);
    return { triples: [], salience: 0.5 };
  }
}

/**
 * Convert a triple to a human-readable string for embedding.
 *
 * HippoRAG stores triples as strings in the fact embedding store
 * so they can be matched against queries via cosine similarity.
 *
 * Example: { subject: "alice", predicate: "works_at", object: "google" }
 *   → "alice works_at google"
 *
 * @param triple — the triple to stringify
 * @returns a space-separated string representation
 */
export function tripleToString(triple: Triple): string {
  return `${triple.subject} ${triple.predicate} ${triple.object}`;
}
