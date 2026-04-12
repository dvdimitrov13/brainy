/**
 * notepad.ts — Structured markdown notepad for long-term memory.
 *
 * Replaces HippoRAG with a simpler approach: the agent maintains a
 * markdown document like a person taking notes. No embeddings, no
 * graph algorithms — just structured text the agent can read, write,
 * and edit through tools.
 *
 * The notepad has:
 *   - Hierarchical sections organized by topic
 *   - Auto-generated table of contents
 *   - Citations linking notes to source exchanges [exchange:sess0-turn3]
 *   - Raw exchange storage for drilling into citations
 *   - Rolling summary for conversational continuity
 *   - Read tracking to prevent blind edits (must read before edit)
 */

import { llmFast } from "../llm.ts";
import {
  generateTOC,
  insertSection,
  updateSection,
  extractSection,
  countSections,
  estimateTokens,
} from "./notepad-ops.ts";

/** A note operation the agent wants to apply */
export interface NoteOperation {
  /** Hierarchical path: "Company/Team" or "Personal/Preferences" */
  sectionPath: string;
  /** The note content (markdown, with [exchange:ID] citations) */
  content: string;
  /** Optional: place this section after the named section */
  afterSection?: string;
}

/** Token threshold for conversation buffer pressure */
const TOKEN_THRESHOLD = 1024;

export class NotepadMemory {
  /** The full markdown notepad (without TOC — TOC is generated on read) */
  private body: string = "";
  /** Raw exchanges stored by ID for recall_exchange tool */
  private exchanges: Map<string, string> = new Map();
  /** Rolling summary of conversation so far */
  private rollingSummary: string = "";
  /** Current session index (incremented at session boundaries) */
  private currentSession: number = 0;
  /** Sections the agent has read in the current buffer cycle */
  private readSections: Set<string> = new Set();

  // ══════════════════════════════════════════════
  // BUFFER MANAGEMENT
  // ══════════════════════════════════════════════

  /** Check if the conversation buffer has exceeded pressure threshold */
  shouldSummarize(buffer: string): boolean {
    return estimateTokens(buffer) > TOKEN_THRESHOLD;
  }

  /** Append a new exchange to the conversation buffer */
  append(currentBuffer: string, newExchange: string): string {
    if (!currentBuffer) return newExchange;
    return `${currentBuffer}\n\n${newExchange}`;
  }

  // ══════════════════════════════════════════════
  // READING
  // ══════════════════════════════════════════════

  /**
   * Get the table of contents.
   * This is always injected into the system prompt so the agent
   * knows what sections exist in the notepad.
   */
  getTOC(): string {
    return generateTOC(this.body) || "(empty notepad)";
  }

  /**
   * Read a section's direct content (no subsections).
   * Marks the section as "read" so edit_notes can modify it.
   */
  readSection(heading: string): string | null {
    const content = extractSection(this.body, heading);
    if (content !== null) {
      this.readSections.add(heading.toLowerCase());
    }
    return content;
  }

  /** Get a raw exchange by citation ID */
  getExchange(id: string): string | null {
    return this.exchanges.get(id) ?? null;
  }

  /** Get the rolling summary */
  getRollingSummary(): string {
    return this.rollingSummary;
  }

  /** Get the full notepad markdown (for debugging/viz) */
  getFullContent(): string {
    return this.body;
  }

  // ══════════════════════════════════════════════
  // WRITING
  // ══════════════════════════════════════════════

  /** Store a raw exchange by ID for future recall */
  storeExchange(id: string, text: string): void {
    this.exchanges.set(id, text);
  }

  /**
   * Apply note operations to the notepad.
   * Each operation inserts a new section at the specified path.
   * TOC is regenerated automatically after all operations.
   */
  writeNotes(operations: NoteOperation[]): void {
    for (const op of operations) {
      this.body = insertSection(
        this.body,
        op.sectionPath,
        op.content,
        op.afterSection
      );
    }
  }

  /**
   * Edit an existing section's content.
   * Returns an error if the section hasn't been read first.
   */
  editSection(
    heading: string,
    content: string
  ): { ok: boolean; error?: string } {
    if (!this.canEdit(heading)) {
      return {
        ok: false,
        error: `Cannot edit "${heading}" — you must read it first with read_notes.`,
      };
    }

    const before = this.body;
    this.body = updateSection(this.body, heading, content);

    if (this.body === before) {
      return {
        ok: false,
        error: `Section "${heading}" not found in the notepad.`,
      };
    }

    return { ok: true };
  }

  /**
   * Generate a rolling summary from the previous summary + conversation buffer.
   * Called after write_notes clears memory pressure.
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
  // READ TRACKING (edit failsafe)
  // ══════════════════════════════════════════════

  /** Check if a section has been read in the current buffer cycle */
  canEdit(section: string): boolean {
    return this.readSections.has(section.toLowerCase());
  }

  /** Reset read tracking (called after buffer compression) */
  resetReadTracking(): void {
    this.readSections.clear();
  }

  // ══════════════════════════════════════════════
  // SESSION MANAGEMENT
  // ══════════════════════════════════════════════

  /** Start a new session (called at session boundaries) */
  newSession(): void {
    this.currentSession++;
  }

  /** Get the current session index */
  getSessionIndex(): number {
    return this.currentSession;
  }

  // ══════════════════════════════════════════════
  // STATS
  // ══════════════════════════════════════════════

  getStats(): {
    notepadTokens: number;
    exchangeCount: number;
    sectionCount: number;
  } {
    return {
      notepadTokens: estimateTokens(this.body),
      exchangeCount: this.exchanges.size,
      sectionCount: countSections(this.body),
    };
  }
}
