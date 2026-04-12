/**
 * state.ts — LangGraph state definition for the Brainy agent.
 *
 * MEMORY MODEL: Notepad-based.
 *
 * The agent maintains a structured markdown notepad for long-term memory.
 * Real conversation turns accumulate in a buffer. When the buffer exceeds
 * ~1024 tokens, the graph loops back to the respond node with a forced
 * write_notes instruction — the agent must process pending exchanges
 * into notes before continuing.
 *
 * The notepad and exchange storage live in a singleton (NotepadMemory),
 * not in this state. State only carries the per-turn and per-session
 * transient fields.
 */

import { Annotation } from "@langchain/langgraph";

/** A pending exchange with its citation ID */
export interface PendingExchange {
  id: string;
  text: string;
}

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

  /**
   * Exchanges not yet written to the notepad.
   * Each entry carries a citation ID (e.g., "sess0-turn3") and the raw text.
   * Cleared after the agent calls write_notes.
   */
  pendingExchanges: Annotation<PendingExchange[]>({
    reducer: (_prev, next) => next,
    default: () => [],
  }),

  /**
   * Flag: the agent MUST call write_notes before responding.
   * Set by the memorize node when buffer exceeds pressure threshold.
   * Cleared by the respond node after write_notes is called.
   */
  mustWriteNotes: Annotation<boolean>({
    reducer: (_prev, next) => next,
    default: () => false,
  }),

  /** Conversation turn counter. */
  turnCount: Annotation<number>({
    reducer: (_prev, next) => next,
    default: () => 0,
  }),
});
