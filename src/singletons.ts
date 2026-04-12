/**
 * singletons.ts — Shared memory system instances.
 *
 * LangGraph state should contain serializable data (strings, numbers),
 * not class instances. So the NotepadMemory lives here as a module-level
 * singleton that graph nodes import directly.
 */

import { NotepadMemory } from "./memory/notepad.ts";

/** The notepad memory system (structured notes + exchange storage) */
export let notepadMemory = new NotepadMemory();

/**
 * Reset memory to a fresh state.
 * Used by the evaluation harness between test questions.
 */
export function resetMemory(): void {
  notepadMemory = new NotepadMemory();
}
