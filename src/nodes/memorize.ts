/**
 * memorize.ts — LangGraph node that manages conversation memory.
 *
 * After the AI has responded, we handle memory in a pressure-based model:
 *
 *   1. Append the new exchange to the conversation buffer
 *   2. Track it as a pending exchange (not yet indexed in HippoRAG)
 *   3. Check if the buffer exceeds the token threshold (~1024 tokens)
 *   4. If YES:
 *      a. Summarize each pending exchange individually (4:1 compression, parallel)
 *      b. Index each summary into HippoRAG (parallel)
 *      c. Replace the buffer with concatenated summaries
 *   5. If NO: just keep accumulating — no LLM calls needed
 *
 * HippoRAG indexes dense summaries (not raw exchanges), keeping passages
 * compact and focused for better triple extraction and retrieval.
 *
 * Graph position: START → retrieve → respond → [memorize] → END
 */

import type { BrainyState } from "../state.ts";
import { hipporag, compactMemory } from "../singletons.ts";

/**
 * Update conversation memory with the new exchange.
 */
export async function memorizeNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const newTurnCount = state.turnCount + 1;

  const exchangeText = `User: ${state.userMessage}\nAssistant: ${state.aiResponse}`;

  let newBuffer = compactMemory.append(
    state.conversationBuffer,
    exchangeText
  );

  let newPending = [...state.pendingExchanges, exchangeText];

  // ── Check memory pressure ──
  // IMPORTANT: This check runs AFTER the full turn is complete (retrieve →
  // respond → memorize). We never cut off the agent mid-turn. If tool use
  // is added later, ensure the agent finishes its entire agentic loop
  // (all tool calls + final response) before checking pressure — don't
  // compress between tool steps.
  if (compactMemory.shouldSummarize(newBuffer)) {
    // Summarize each pending exchange individually (4:1 compression)
    const summaries = await compactMemory.summarizeExchanges(newPending);

    // Index each summary into HippoRAG (parallel)
    await Promise.all(summaries.map((summary) => hipporag.index(summary)));

    // Replace buffer with concatenated summaries
    newBuffer = "[Summary of earlier conversation]\n" + summaries.join("\n\n");
    newPending = [];

    hipporag.forget();
  }

  return {
    conversationBuffer: newBuffer,
    pendingExchanges: newPending,
    turnCount: newTurnCount,
  };
}
