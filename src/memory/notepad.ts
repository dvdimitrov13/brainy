/**
 * notepad.ts — Flat chronological notepad for long-term memory.
 *
 * The notepad is a flat list of self-contained notes, each with:
 *   - A date (when the fact occurred)
 *   - Dense content (the fact, with enough context to be findable)
 *   - Citation(s) to source exchanges
 *
 * No hierarchy — each note stands alone. A fact like "Viewed Cedar Creek
 * property — out of budget" is findable whether you're looking for
 * "properties viewed", "budget", or "Cedar Creek".
 *
 * The agent can:
 *   - write_notes: add new flat entries
 *   - read_notes: get full notepad or search by keyword
 *   - edit_notes: update a note by index (must read first)
 *   - recall_exchange: fetch raw exchange from citation
 */

import { llmFast } from "../llm.ts";
import type { NoteEntry } from "./notepad-ops.ts";
import {
  formatNotepad,
  generateIndex,
  searchNotes,
  estimateTokens,
} from "./notepad-ops.ts";

/** Token threshold for conversation buffer pressure */
const TOKEN_THRESHOLD = 1024;

export class NotepadMemory {
  /** Flat chronological list of notes */
  private notes: NoteEntry[] = [];
  /** Raw exchanges stored by ID for recall */
  private exchanges: Map<string, string> = new Map();
  /** Rolling summary of conversation */
  private rollingSummary: string = "";
  /** Current session index */
  private currentSession: number = 0;
  /** Note indices that have been read (for edit failsafe) */
  private readNoteIndices: Set<number> = new Set();

  // ══════════════════════════════════════════════
  // BUFFER MANAGEMENT
  // ══════════════════════════════════════════════

  shouldSummarize(buffer: string): boolean {
    return estimateTokens(buffer) > TOKEN_THRESHOLD;
  }

  append(currentBuffer: string, newExchange: string): string {
    if (!currentBuffer) return newExchange;
    return `${currentBuffer}\n\n${newExchange}`;
  }

  // ══════════════════════════════════════════════
  // READING
  // ══════════════════════════════════════════════

  /**
   * Get the notepad index — a compact scannable summary.
   * Always injected into the system prompt.
   */
  getIndex(): string {
    return generateIndex(this.notes);
  }

  /**
   * Get all notes formatted as a readable list.
   * Marks all notes as "read" for edit purposes.
   */
  getAllNotes(): string {
    for (let i = 0; i < this.notes.length; i++) {
      this.readNoteIndices.add(i);
    }
    return formatNotepad(this.notes);
  }

  /**
   * Search notes by keyword query.
   * Returns matching notes. Marks them as read.
   */
  search(query: string): string {
    const results = searchNotes(this.notes, query);
    if (results.length === 0) return "No matching notes found.";

    for (const { index } of results) {
      this.readNoteIndices.add(index);
    }

    return results
      .map(
        ({ index, note }) =>
          `${index + 1}. [${note.date}] ${note.content} ${note.citations.map((c) => `[exchange:${c}]`).join(" ")}`
      )
      .join("\n");
  }

  /** Get a raw exchange by citation ID */
  getExchange(id: string): string | null {
    return this.exchanges.get(id) ?? null;
  }

  /** Get the rolling summary */
  getRollingSummary(): string {
    return this.rollingSummary;
  }

  /** Get full notepad content (for debugging/viz) */
  getFullContent(): string {
    return formatNotepad(this.notes);
  }

  // ══════════════════════════════════════════════
  // WRITING
  // ══════════════════════════════════════════════

  /** Store a raw exchange by ID */
  storeExchange(id: string, text: string): void {
    this.exchanges.set(id, text);
  }

  /**
   * Add new notes to the notepad.
   * Notes are appended chronologically.
   */
  writeNotes(entries: NoteEntry[]): void {
    this.notes.push(...entries);
    // Sort by date for chronological order
    this.notes.sort((a, b) => a.date.localeCompare(b.date));
  }

  /**
   * Edit an existing note by index (1-based).
   * Must have been read first via getAllNotes or search.
   */
  editNote(
    index: number,
    content: string
  ): { ok: boolean; error?: string } {
    const idx = index - 1; // convert to 0-based
    if (idx < 0 || idx >= this.notes.length) {
      return { ok: false, error: `Note #${index} does not exist.` };
    }
    if (!this.readNoteIndices.has(idx)) {
      return {
        ok: false,
        error: `Cannot edit note #${index} — you must read it first with read_notes.`,
      };
    }
    this.notes[idx]!.content = content;
    return { ok: true };
  }

  /**
   * Generate rolling summary from previous summary + conversation buffer.
   */
  async updateRollingSummary(buffer: string): Promise<void> {
    const response = await llmFast.invoke([
      {
        role: "system" as const,
        content: `Compress conversation context into a rolling summary (2-4 sentences).

Previous summary: ${this.rollingSummary || "(none — first compression)"}

Recent conversation:
${buffer}

Capture the thread: topics discussed, key facts, where the conversation is heading.
Details are stored in the notepad — this just provides context for the next turns.

Respond with ONLY the summary.`,
      },
      {
        role: "user" as const,
        content: "Generate the rolling summary.",
      },
    ]);

    this.rollingSummary =
      typeof response.content === "string"
        ? response.content.trim()
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((b) => b.type === "text")
            .map((b) => b.text ?? "")
            .join("")
            .trim();
  }

  // ══════════════════════════════════════════════
  // READ TRACKING
  // ══════════════════════════════════════════════

  resetReadTracking(): void {
    this.readNoteIndices.clear();
  }

  // ══════════════════════════════════════════════
  // SESSION MANAGEMENT
  // ══════════════════════════════════════════════

  newSession(): void {
    this.currentSession++;
  }

  getSessionIndex(): number {
    return this.currentSession;
  }

  // ══════════════════════════════════════════════
  // STATS
  // ══════════════════════════════════════════════

  getStats(): {
    notepadTokens: number;
    exchangeCount: number;
    noteCount: number;
  } {
    const content = formatNotepad(this.notes);
    return {
      notepadTokens: estimateTokens(content),
      exchangeCount: this.exchanges.size,
      noteCount: this.notes.length,
    };
  }
}
