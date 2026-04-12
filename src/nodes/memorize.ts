/**
 * memorize.ts — LangGraph node that manages conversation memory.
 *
 * Pressure-based model:
 *   1. Append exchange to buffer + pending list
 *   2. If buffer > 1024 tokens: index each pending exchange into HippoRAG
 *      (processExchange produces summary + triples + tags in one LLM call)
 *   3. Replace buffer with rolling summary
 *   4. Session boundary: flush pending, reset buffer
 *
 * Graph position: START → respond → [memorize] → END
 */

import type { BrainyState } from "../state.ts";
import { hipporag } from "../singletons.ts";
import { llmFast } from "../llm.ts";

/** Token threshold for buffer pressure */
const TOKEN_THRESHOLD = 1024;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Generate rolling summary from previous summary + buffer */
async function rollingSummary(
  prevSummary: string,
  buffer: string
): Promise<string> {
  const response = await llmFast.invoke([
    {
      role: "system" as const,
      content: `Compress conversation context into a rolling summary (2-4 sentences).

Previous summary: ${prevSummary || "(none)"}

Recent conversation:
${buffer}

Capture the thread: topics discussed, key facts, where the conversation is heading.
Details are stored in long-term memory — this just provides context.

Respond with ONLY the summary.`,
    },
    {
      role: "user" as const,
      content: "Generate the rolling summary.",
    },
  ]);

  return typeof response.content === "string"
    ? response.content.trim()
    : (response.content as Array<{ type: string; text?: string }>)
        .filter((b) => b.type === "text")
        .map((b) => b.text ?? "")
        .join("")
        .trim();
}

export async function memorizeNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const newTurnCount = state.turnCount + 1;

  const exchangeText = `User: ${state.userMessage}\nAssistant: ${state.aiResponse}`;

  let newBuffer = state.conversationBuffer
    ? `${state.conversationBuffer}\n\n${exchangeText}`
    : exchangeText;

  let newPending = [...state.pendingExchanges, exchangeText];

  // ── Check memory pressure ──
  // IMPORTANT: This check runs AFTER the full turn is complete.
  if (estimateTokens(newBuffer) > TOKEN_THRESHOLD && newPending.length > 0) {
    // Index each pending exchange into HippoRAG (parallel).
    // Each call: summarize + extract triples + assign tags in one LLM call,
    // then embed + add to knowledge graph.
    await Promise.all(newPending.map((ex) => hipporag.index(ex)));

    // Generate rolling summary and compress buffer
    // Extract current summary if buffer starts with [Summary]
    const prevSummary = newBuffer.startsWith("[Summary]")
      ? newBuffer.slice("[Summary]\n".length).split("\n\n")[0] ?? ""
      : "";

    const summary = await rollingSummary(prevSummary, newBuffer);
    newBuffer = `[Summary]\n${summary}`;
    newPending = [];

    hipporag.forget();
  }

  return {
    conversationBuffer: newBuffer,
    pendingExchanges: newPending,
    turnCount: newTurnCount,
  };
}
