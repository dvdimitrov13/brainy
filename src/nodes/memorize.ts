/**
 * memorize.ts — LangGraph node that updates both memory stores.
 *
 * This is the final node in the pipeline. After the AI has responded,
 * we store the exchange in both memory systems:
 *
 *   1. Compact Memory: update the one-sentence summary to include
 *      the new exchange.
 *
 *   2. HippoRAG2 Vector Memory: index the exchange — extract triples,
 *      create embeddings, add to the knowledge graph.
 *
 * Both operations run in PARALLEL (Promise.all) since they're independent.
 * This saves ~1-2 seconds per turn since each involves an LLM call.
 *
 * Every 10 turns, we also run the "summary of summaries" to correct
 * drift in the compact memory.
 *
 * Graph position: START → retrieve → respond → [memorize] → END
 *
 * TS note for Python devs:
 *   `Promise.all([a, b])` is like `asyncio.gather(a, b)` — it runs
 *   multiple async operations concurrently and waits for all to finish.
 *   The `[result1, result2] = await Promise.all(...)` syntax is called
 *   "destructuring assignment" — it's like `result1, result2 = await gather(...)`.
 */

import type { BrainyState } from "../state.ts";
import { hipporag, compactMemory } from "../singletons.ts";

/** How often to run the summary-of-summaries (every N turns) */
const SUMMARY_OF_SUMMARIES_EVERY = 10;

/**
 * Update both memory stores with the new conversation exchange.
 *
 * @param state — current graph state with userMessage + aiResponse
 * @returns partial state update with new compactSummary, turnCount, etc.
 */
export async function memorizeNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const newTurnCount = state.turnCount + 1;

  // Format the exchange as "User: ... \n Assistant: ..."
  // This is what gets stored as a passage in HippoRAG and summarised
  const exchangeText = `User: ${state.userMessage}\nAssistant: ${state.aiResponse}`;

  // ── Run compact memory update + HippoRAG indexing in parallel ──
  // These are independent operations, so running them concurrently
  // saves time (each involves at least one LLM API call).
  const [newSummary] = await Promise.all([
    // Update compact memory: integrate new exchange into the running summary
    compactMemory.update(state.compactSummary, exchangeText),

    // Index in HippoRAG: extract triples, embed, add to knowledge graph
    hipporag.index(exchangeText),
  ]);

  // ── Periodically run summary-of-summaries ──
  // Every N turns, re-derive the summary from recent summary history
  // to correct for drift (the "two-level" mechanism from the paper)
  let newMetaSummary = state.metaSummary;
  if (newTurnCount % SUMMARY_OF_SUMMARIES_EVERY === 0) {
    newMetaSummary = await compactMemory.generateMetaSummary(newSummary);
  }

  // ── Run semantic forgetting ──
  // Prune old, low-salience passages if we've exceeded capacity.
  // This is fast (no API calls), just scoring and removing.
  hipporag.forget();

  return {
    compactSummary: newSummary,
    turnCount: newTurnCount,
    metaSummary: newMetaSummary,
  };
}
