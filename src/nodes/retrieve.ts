/**
 * retrieve.ts — LangGraph node that auto-recognizes associations every turn.
 *
 * Flow:
 *   1. Haiku contextualizes the user message using rolling summary + buffer
 *   2. HippoRAG.recognize() finds matching triples
 *   3. Triples stored in state for the respond node to see
 *
 * This runs automatically every turn. The agent also has a recognize tool
 * to search with custom queries.
 *
 * Graph position: START → [retrieve] → respond → memorize → END
 */

import type { BrainyState } from "../state.ts";
import { hipporag } from "../singletons.ts";
import { llmFast } from "../llm.ts";

/**
 * Contextualize the user's message using conversation history.
 *
 * Haiku rewrites the message as a focused search query that incorporates
 * context from the rolling summary and recent conversation. This ensures
 * recognize finds the right associations even for vague messages like
 * "how did that go?" or "what about the other one?".
 */
async function contextualizeQuery(
  userMessage: string,
  buffer: string
): Promise<string> {
  // If no buffer context, the message is already self-contained
  if (!buffer) return userMessage;

  const response = await llmFast.invoke([
    {
      role: "system" as const,
      content: `Rewrite the user's message as a self-contained search query that incorporates context from the conversation.

Examples:
- "how did that go?" + context about 5K race → "user's 5K race result and experience"
- "what about the price?" + context about Brookside townhouse → "Brookside townhouse purchase price"
- "tell me more" + context about kitchen renovation → "kitchen renovation details and countertop selection"

If the message is already self-contained, return it as-is.
Respond with ONLY the rewritten query, nothing else.`,
    },
    {
      role: "user" as const,
      content: `Conversation context:\n${buffer}\n\nUser message: ${userMessage}`,
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

/**
 * Auto-recognize: contextualize query → find matching triples.
 */
export async function retrieveNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  if (!state.userMessage) {
    return { recognizedTriples: "" };
  }

  // Skip auto-recognize if no passages indexed yet
  const stats = hipporag.getStats();
  if (stats.passages === 0) {
    return { recognizedTriples: "" };
  }

  // Contextualize the query using conversation history
  const contextualizedQuery = await contextualizeQuery(
    state.userMessage,
    state.conversationBuffer
  );

  // Recognize: find matching triples
  const triples = await hipporag.recognize(contextualizedQuery);

  // Format triples for injection into the system prompt
  let recognizedTriples = "";
  if (triples.length > 0) {
    recognizedTriples = triples
      .map(
        (t, i) =>
          `${i + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
      )
      .join("\n");
  }

  return { recognizedTriples };
}
