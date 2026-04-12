/**
 * notepad-ops.ts — Pure operations for the flat notepad.
 *
 * The notepad is a flat chronological list of self-contained notes.
 * Each note has a date, content, and citation. No hierarchy — each
 * note stands on its own with enough context to be found from any angle.
 *
 * The index is auto-generated from note content, providing a scannable
 * summary of what's in the notepad.
 */

/** A single note entry */
export interface NoteEntry {
  /** When this fact occurred (from session date or conversation context) */
  date: string;
  /** The note content — self-contained, dense, includes relevant context */
  content: string;
  /** Source exchange citation(s) */
  citations: string[];
}

/** Token estimation (~4 chars per token) */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Format notes into a readable markdown document.
 */
export function formatNotepad(notes: NoteEntry[]): string {
  if (notes.length === 0) return "(empty notepad)";

  return notes
    .map(
      (n, i) =>
        `${i + 1}. [${n.date}] ${n.content} ${n.citations.map((c) => `[exchange:${c}]`).join(" ")}`
    )
    .join("\n");
}

/**
 * Generate a compact index/summary from the notes.
 *
 * Groups notes by date and shows a brief preview of each,
 * giving the agent a scannable overview to decide what to read.
 */
export function generateIndex(notes: NoteEntry[]): string {
  if (notes.length === 0) return "(empty notepad)";

  // Group by date
  const byDate = new Map<string, NoteEntry[]>();
  for (const note of notes) {
    const dateKey = note.date || "undated";
    if (!byDate.has(dateKey)) byDate.set(dateKey, []);
    byDate.get(dateKey)!.push(note);
  }

  const lines: string[] = [];
  for (const [date, dateNotes] of byDate) {
    const previews = dateNotes
      .map((n) => n.content.slice(0, 60) + (n.content.length > 60 ? "..." : ""))
      .join("; ");
    lines.push(`[${date}] ${previews}`);
  }

  return lines.join("\n");
}

/**
 * Search notes by keyword (case-insensitive).
 * Returns matching notes with their indices.
 */
export function searchNotes(
  notes: NoteEntry[],
  query: string
): { index: number; note: NoteEntry }[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [];

  return notes
    .map((note, index) => ({ index, note }))
    .filter(({ note }) => {
      const text = `${note.date} ${note.content}`.toLowerCase();
      return terms.some((term) => text.includes(term));
    });
}
