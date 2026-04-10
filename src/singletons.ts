/**
 * singletons.ts — Shared instances of memory systems.
 *
 * LangGraph state should contain serializable data (strings, numbers),
 * not class instances with methods. So we keep the memory system
 * instances here as module-level singletons that the graph nodes
 * import directly.
 *
 * Python analogy: these are like module-level global variables.
 * In JS/TS, each module is executed exactly once, so exporting a
 * `const` here guarantees a single shared instance.
 *
 * Both nodes/retrieve.ts and nodes/memorize.ts import from here
 * to ensure they're working with the same memory stores.
 *
 * The `reset()` function creates fresh instances — needed by the
 * evaluation harness to clear state between test questions.
 */

import { HippoRAG } from "./hipporag/index.ts";
import { CompactMemory } from "./memory/compact-memory.ts";

/** The HippoRAG2 vector memory system (knowledge graph + PPR retrieval) */
export let hipporag = new HippoRAG();

/** The compact memory system (one-sentence summary + summary-of-summaries) */
export let compactMemory = new CompactMemory();

/**
 * Reset both memory systems to a fresh state.
 *
 * Used by the evaluation harness between test questions so each
 * question starts with a clean knowledge graph and summary.
 */
export function resetMemory(): void {
  hipporag = new HippoRAG();
  compactMemory = new CompactMemory();
}
