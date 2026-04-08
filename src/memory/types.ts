/**
 * types.ts — Shared types for the memory subsystem.
 *
 * These types are used by both the compact memory and the main
 * graph nodes to communicate retrieved context.
 */

/**
 * The result of a memory retrieval operation.
 *
 * Combines context from both memory systems:
 *   - Compact memory: the narrative summary
 *   - Vector memory (HippoRAG): specific relevant passages
 */
export interface MemoryContext {
  /** The one-sentence compact summary of the conversation so far */
  compactSummary: string;
  /** The second-level meta summary (if available) */
  metaSummary: string;
  /** Formatted string of relevant passages from HippoRAG */
  retrievedPassages: string;
}
