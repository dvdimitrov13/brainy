/**
 * memorize.ts — LangGraph node that manages conversation memory.
 *
 * This is the final node in the pipeline. After the AI has responded,
 * we handle memory in a pressure-based model:
 *
 *   1. Append the new exchange to the conversation buffer
 *   2. Track it as a pending exchange (not yet indexed in HippoRAG)
 *   3. Check if the buffer exceeds the token threshold (~1024 tokens)
 *   4. If YES: index each pending exchange separately into HippoRAG,
 *      then summarize the buffer and clear the pending list
 *   5. If NO: just keep accumulating — no LLM calls needed
 *
 * Each exchange is indexed as a separate passage in HippoRAG so that
 * triple extraction stays focused and retrieval returns granular,
 * relevant passages — not one big blob of merged conversation.
 *
 * Graph position: START → retrieve → respond → [memorize] → END
 */

import type { BrainyState } from "../state.ts";
import { hipporag, compactMemory } from "../singletons.ts";

/**
 * Update conversation memory with the new exchange.
 *
 * @param state — current graph state with userMessage + aiResponse
 * @returns partial state update with updated conversationBuffer, pendingExchanges, and turnCount
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

  // Track this exchange as pending (not yet indexed in HippoRAG)
  let newPending = [...state.pendingExchanges, exchangeText];

  // ── Check memory pressure ──
  // IMPORTANT: This check runs AFTER the full turn is complete (retrieve →
  // respond → memorize). We never cut off the agent mid-turn. If tool use
  // is added later, ensure the agent finishes its entire agentic loop
  // (all tool calls + final response) before checking pressure — don't
  // compress between tool steps.
  if (compactMemory.shouldSummarize(newBuffer)) {
    // Buffer exceeded threshold — index each exchange separately, then compress.

    // Index each pending exchange as a separate passage in HippoRAG.
    // This keeps passages granular: each one gets its own triples extracted,
    // its own embedding, and shows up as a distinct node in the knowledge graph.
    // Summarize the buffer in parallel with indexing.
    const [summary] = await Promise.all([
      compactMemory.summarize(newBuffer),
      ...newPending.map((exchange) => hipporag.index(exchange)),
    ]);

    newBuffer = summary;
    newPending = [];

    // Run semantic forgetting — prune old, low-salience passages
    // if we've exceeded capacity. Fast (no API calls).
    hipporag.forget();
  }

  return {
    conversationBuffer: newBuffer,
    pendingExchanges: newPending,
    turnCount: newTurnCount,
  };
}
