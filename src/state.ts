/**
 * state.ts — LangGraph state definition for the Brainy agent.
 *
 * MEMORY MODEL: HippoRAG with tag-based filtering.
 *
 * Real conversation turns accumulate in a buffer. When the buffer exceeds
 * ~1024 tokens, the memorize node summarizes + tags + indexes each pending
 * exchange into HippoRAG. The agent retrieves via a `remember` tool that
 * accepts optional type/topic tag filters.
 */

import { Annotation } from "@langchain/langgraph";

export const BrainyState = Annotation.Root({
  /** The current user message (overwritten each turn). */
  userMessage: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /** The AI's response for the current turn. */
  aiResponse: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /** Conversation buffer — real turns, compressed on pressure. */
  conversationBuffer: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /** Exchanges not yet indexed into HippoRAG. */
  pendingExchanges: Annotation<string[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),

  /** Auto-recognized triples (from contextualized query each turn).
   *  Injected into the system prompt so the agent sees associations. */
  recognizedTriples: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),

  /** Conversation turn counter. */
  turnCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
});
