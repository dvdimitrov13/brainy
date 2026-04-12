/**
 * memorize.ts — LangGraph node that manages conversation memory.
 *
 * This is the final node in the pipeline. After the AI has responded,
 * we handle memory in a pressure-based model:
 *
 *   1. Append the new exchange to the conversation buffer
 *   2. Check if the buffer exceeds the token threshold (~1024 tokens)
 *   3. If YES: summarize the buffer + index the summary into HippoRAG
 *   4. If NO: just keep accumulating — no LLM calls needed
 *
 * HippoRAG only gets indexed at summarization time, meaning it receives
 * denser, multi-turn summaries rather than individual turn pairs. This
 * creates higher-quality knowledge graph entries.
 *
 * Graph position: START → retrieve → respond → [memorize] → END
 *
 * TS note for Python devs:
 *   `Promise.all([a, b])` is like `asyncio.gather(a, b)` — it runs
 *   multiple async operations concurrently and waits for all to finish.
 */

import type { BrainyState } from "../state.ts";
import { hipporag, compactMemory } from "../singletons.ts";

/**
 * Update conversation memory with the new exchange.
 *
 * @param state — current graph state with userMessage + aiResponse
 * @returns partial state update with updated conversationBuffer and turnCount
 */
export async function memorizeNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const newTurnCount = state.turnCount + 1;

  // Format the exchange as "User: ... \n Assistant: ..."
  const exchangeText = `User: ${state.userMessage}\nAssistant: ${state.aiResponse}`;

  // ── Append the new exchange to the buffer ──
  let newBuffer = compactMemory.append(
    state.conversationBuffer,
    exchangeText
  );

  // ── Check memory pressure ──
  // IMPORTANT: This check runs AFTER the full turn is complete (retrieve →
  // respond → memorize). We never cut off the agent mid-turn. If tool use
  // is added later, ensure the agent finishes its entire agentic loop
  // (all tool calls + final response) before checking pressure — don't
  // compress between tool steps.
  if (compactMemory.shouldSummarize(newBuffer)) {
    // Buffer exceeded threshold — compress and index

    // Summarize the buffer and index it into HippoRAG in parallel.
    // The summary goes into long-term memory (knowledge graph),
    // while the compressed text replaces the buffer.
    const [summary] = await Promise.all([
      compactMemory.summarize(newBuffer),
      hipporag.index(newBuffer),
    ]);

    newBuffer = summary;

    // Run semantic forgetting — prune old, low-salience passages
    // if we've exceeded capacity. Fast (no API calls).
    hipporag.forget();
  }

  return {
    conversationBuffer: newBuffer,
    turnCount: newTurnCount,
  };
}
