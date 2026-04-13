/**
 * retrieve.ts — LangGraph node that auto-recognizes associations every turn.
 *
 * Runs recognize directly on the user's message — no contextualization.
 * The agent can call recognize again with a refined query if needed.
 *
 * Graph position: START → [retrieve] → respond → memorize → END
 */

import type { BrainyState } from "../state.ts";
import { hipporag } from "../singletons.ts";

export async function retrieveNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  if (!state.userMessage) {
    return { recognizedTriples: "" };
  }

  const stats = hipporag.getStats();
  if (stats.passages === 0) {
    return { recognizedTriples: "" };
  }

  const triples = await hipporag.recognize(state.userMessage);

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
