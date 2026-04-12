/**
 * singletons.ts — Shared memory system instances.
 */

import { HippoRAG } from "./hipporag/index.ts";

/** The HippoRAG2 memory system with tag-based filtering */
export let hipporag = new HippoRAG();

/**
 * Reset memory to a fresh state.
 * Used by the evaluation harness between test questions.
 */
export function resetMemory(): void {
  hipporag = new HippoRAG();
}
