/**
 * state.ts — LangGraph state definition for the Brainy agent.
 *
 * This defines the "shape" of data that flows between graph nodes.
 * In LangGraph, state is the central concept: every node reads from
 * state and returns partial updates to it.
 *
 * CRITICAL DESIGN DECISION: No message accumulation!
 *
 * Unlike a typical chatbot that accumulates all messages in state,
 * this agent always stays on "turn 1". Each invocation only has:
 *   - The current user message
 *   - The current AI response
 *   - Memory context injected from the two memory stores
 *
 * ALL continuity comes from the memory stores (compact summary +
 * HippoRAG vector memory), NOT from message history. This is what
 * keeps the prompt under ~3.5K tokens regardless of conversation length.
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
   * This is the ONLY user input — no history accumulates.
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
   * The one-sentence compact summary of the entire conversation.
   * Updated by the memorize node after each turn.
   */
  compactSummary: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /**
   * The second-level "summary of summaries".
   * Updated every ~10 turns to correct for drift in incremental summarisation.
   */
  metaSummary: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /**
   * Formatted string of relevant passages retrieved from HippoRAG.
   * Rebuilt each turn during the retrieve node — purely transient.
   */
  retrievedContext: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /**
   * Conversation turn counter.
   * Used to schedule the periodic summary-of-summaries (every 10 turns).
   */
  turnCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
});
