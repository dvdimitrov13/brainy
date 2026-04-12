/**
 * state.ts — LangGraph state definition for the Brainy agent.
 *
 * This defines the "shape" of data that flows between graph nodes.
 * In LangGraph, state is the central concept: every node reads from
 * state and returns partial updates to it.
 *
 * CRITICAL DESIGN DECISION: Pressure-based memory!
 *
 * Instead of always summarizing after every turn (lossy from turn 1),
 * we accumulate real conversation turns in a buffer. The LLM sees
 * actual messages until the buffer exceeds a token threshold (~1024
 * tokens), at which point we:
 *   1. Summarize the buffer into a compact paragraph
 *   2. Index the summary into HippoRAG for long-term retrieval
 *   3. Replace the buffer with the summary
 *
 * This means early turns get full fidelity, and compression only
 * happens when memory pressure forces it — matching how human
 * short-term memory works (you remember recent events in detail,
 * older ones as gist).
 *
 * TS note for Python devs:
 *   `Annotation.Root({ ... })` is LangGraph's way of defining a typed
 *   state schema. It's similar to defining a Pydantic BaseModel in Python
 *   LangGraph. Each field can have:
 *     - A type parameter: `Annotation<string>` = field of type string
 *     - A `reducer`: how to merge updates (default: last-write-wins)
 *     - A `default`: initial value factory
 *
 *   `typeof BrainyState.State` gives you the TypeScript type of the state,
 *   which you use for node function signatures. It's like `TypedDict` in Python.
 */

import { Annotation } from "@langchain/langgraph";

export const BrainyState = Annotation.Root({
  /**
   * The current user message (overwritten each turn).
   * This is the ONLY user input — no history accumulates here.
   */
  userMessage: Annotation<string>({
    reducer: (_prev, next) => next, // always overwrite
    default: () => "",
  }),

  /**
   * The AI's response for the current turn (overwritten each turn).
   */
  aiResponse: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /**
   * The conversation buffer — real turns or a compressed summary.
   *
   * Starts empty, accumulates "User: ...\nAssistant: ..." entries.
   * When the buffer exceeds TOKEN_THRESHOLD (~1024 tokens), the
   * memorize node summarizes it and replaces the contents with
   * a compact summary paragraph. This summary then becomes the
   * base that new turns accumulate on top of, until the next
   * compression cycle.
   *
   * The LLM always sees this buffer as its conversation context,
   * so it gets full-fidelity recent turns mixed with compressed
   * older history.
   */
  conversationBuffer: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /**
   * Exchanges accumulated since the last HippoRAG indexing.
   *
   * Each entry is a "User: ...\nAssistant: ..." string. When memory
   * pressure triggers, each exchange is indexed into HippoRAG as a
   * separate passage (not the whole buffer as one blob). This keeps
   * passages granular for better triple extraction and retrieval.
   *
   * After indexing, this array is cleared.
   */
  pendingExchanges: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),

  /**
   * Conversation turn counter.
   */
  turnCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
});
