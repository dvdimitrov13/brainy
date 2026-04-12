/**
 * memorize.ts — LangGraph node that manages conversation memory.
 *
 * After the AI has responded, this node:
 *   1. Stores the raw exchange by ID (for recall_exchange citations)
 *   2. Appends the exchange to the conversation buffer
 *   3. Checks memory pressure (buffer > 1024 tokens)
 *   4. If pressure AND pending exchanges: sets mustWriteNotes flag
 *      → graph loops back to respond for forced write_notes
 *   5. If mustWriteNotes was just cleared (agent wrote notes):
 *      generates rolling summary, compresses buffer, clears pending
 *
 * Graph position: START → respond → [memorize] → END or → respond (pressure)
 */

import type { BrainyState } from "../state.ts";
import type { PendingExchange } from "../state.ts";
import { notepadMemory } from "../singletons.ts";

/**
 * Update conversation memory with the new exchange.
 */
export async function memorizeNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  // ── If we're returning from a forced write_notes cycle ──
  // The agent already wrote notes. Now we compress the buffer.
  if (
    state.mustWriteNotes === false &&
    state.pendingExchanges.length === 0 &&
    state.conversationBuffer
  ) {
    // This means we just cleared mustWriteNotes in respond.
    // Nothing to do — buffer was already handled.
    return {};
  }

  const newTurnCount = state.turnCount + 1;

  // Format the exchange
  const exchangeText = `User: ${state.userMessage}\nAssistant: ${state.aiResponse}`;

  // Generate citation ID
  const exchangeId = `sess${notepadMemory.getSessionIndex()}-turn${state.turnCount}`;

  // Store the raw exchange for recall_exchange tool
  notepadMemory.storeExchange(exchangeId, exchangeText);

  // Append to buffer
  let newBuffer = notepadMemory.append(
    state.conversationBuffer,
    exchangeText
  );

  // Track as pending (not yet written to notepad)
  const newPending: PendingExchange[] = [
    ...state.pendingExchanges,
    { id: exchangeId, text: exchangeText },
  ];

  // ── Check memory pressure ──
  // IMPORTANT: This check runs AFTER the full turn is complete.
  // If pressure exceeds and we have pending exchanges, set the flag
  // to force the agent to call write_notes on the next respond cycle.
  if (
    notepadMemory.shouldSummarize(newBuffer) &&
    newPending.length > 0 &&
    !state.mustWriteNotes
  ) {
    return {
      conversationBuffer: newBuffer,
      pendingExchanges: newPending,
      turnCount: newTurnCount,
      mustWriteNotes: true,
    };
  }

  return {
    conversationBuffer: newBuffer,
    pendingExchanges: newPending,
    turnCount: newTurnCount,
  };
}
