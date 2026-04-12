/**
 * retrieve.ts — LangGraph node that surfaces relevant associations.
 *
 * This runs Phase 1 of HippoRAG retrieval: triple matching + recognition
 * memory filtering. The result is a lightweight set of entity associations
 * (triples) that the agent sees every turn.
 *
 * The agent does NOT get full passages here — those require actively
 * calling the "recall" tool (Phase 2: PPR over the knowledge graph).
 * This mirrors how memory works: associations surface automatically,
 * but recalling the full context takes deliberate effort.
 *
 * Graph position: START → [retrieve] → respond → memorize → END
 */

import type { BrainyState } from "../state.ts";
import { hipporag } from "../singletons.ts";
import { tripleToString } from "../hipporag/openie.ts";

/**
 * Retrieve relevant memory associations for the current user message.
 *
 * @param state — current graph state (contains userMessage)
 * @returns partial state update with retrievedTriples populated
 */
export async function retrieveNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const query = state.userMessage;

  if (!query) {
    return { retrievedTriples: "" };
  }

  // Phase 1: get filtered triples (fast, no PPR)
  const triples = await hipporag.retrieveTriples(query);

  // Format triples as a readable string for the agent
  let retrievedTriples = "";
  if (triples.length > 0) {
    retrievedTriples = triples
      .map(
        (t, i) =>
          `${i + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
      )
      .join("\n");
  }

  return { retrievedTriples };
}
