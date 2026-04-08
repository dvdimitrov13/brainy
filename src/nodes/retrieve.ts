/**
 * retrieve.ts — LangGraph node that queries both memory stores.
 *
 * This is the first node in the pipeline. It takes the user's message
 * and retrieves relevant context from:
 *   1. HippoRAG2 vector memory (episodic recall via PPR)
 *   2. Compact memory (already in state — no action needed)
 *
 * The retrieved passages are formatted into a string and stored in
 * state.retrievedContext, which the respond node will inject into
 * the LLM prompt.
 *
 * Graph position: START → [retrieve] → respond → memorize → END
 *
 * TS note for Python devs:
 *   - `typeof BrainyState.State` is the TypeScript type of the state object.
 *     It's like type-hinting a function parameter as `state: BrainyStateDict`.
 *   - `Partial<typeof BrainyState.State>` means "an object with some (not all)
 *     of the state's fields". Nodes return partial updates — LangGraph merges
 *     them into the full state using the reducers.
 */

import type { BrainyState } from "../state.ts";
import { hipporag } from "../singletons.ts";

/**
 * Retrieve relevant memories for the current user message.
 *
 * @param state — current graph state (contains userMessage, compactSummary, etc.)
 * @returns partial state update with retrievedContext populated
 */
export async function retrieveNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const query = state.userMessage;

  // Skip retrieval if no message or no indexed passages yet
  if (!query) {
    return { retrievedContext: "" };
  }

  // Query HippoRAG2 for relevant passages
  // This runs the full pipeline: dense retrieval → recognition memory → PPR
  const passages = await hipporag.retrieve(query, 3);

  // Format retrieved passages into a readable context string
  let retrievedContext = "";
  if (passages.length > 0) {
    retrievedContext = passages
      .map(
        (passage, i) =>
          `[Memory ${i + 1}]: ${passage.text}`
      )
      .join("\n\n");
  }

  return { retrievedContext };
}
