/**
 * compact-memory.ts — Continuously updated one-sentence conversation summary.
 *
 * This implements the "Compact Memory" from the paper — a single sentence
 * that captures the global narrative thread of the conversation. It's
 * updated after every turn by asking the LLM to integrate the new exchange
 * into the existing summary.
 *
 * The key challenge with incremental summarisation is "cascade errors" —
 * small inaccuracies compound over hundreds of turns (like the telephone
 * game). The paper solves this with a "two-level summary-of-summaries":
 * periodically, instead of updating the summary incrementally, we
 * re-derive it from a buffer of recent summaries. This corrects drift.
 *
 * Brain analogy: Compact Memory is like the brain's "gist memory" —
 * you remember the general storyline of a movie even if you forget
 * specific scenes.
 *
 * TS note for Python devs:
 *   - `private` fields like `summaryHistory` are class-level variables
 *     that can't be accessed outside the class (Python uses _ prefix).
 *   - `async/await` works identically to Python's asyncio.
 */

import { llm } from "../llm.ts";

export class CompactMemory {
  /**
   * Rolling buffer of past summaries.
   *
   * Every time `update()` is called, the new summary is pushed here.
   * This buffer is used by `generateMetaSummary()` to re-derive a
   * fresh summary, correcting any drift from incremental updates.
   *
   * Python equivalent: list[str]
   */
  private summaryHistory: string[] = [];

  /**
   * Update the compact summary with a new conversation exchange.
   *
   * Takes the current summary and the new user+assistant exchange,
   * then asks the LLM to produce an updated one-sentence summary
   * that integrates the new information.
   *
   * @param currentSummary — the existing one-sentence summary (or "" if first turn)
   * @param newExchange    — the new conversation text to integrate
   * @returns the updated one-sentence summary
   */
  async update(currentSummary: string, newExchange: string): Promise<string> {
    const response = await llm.invoke([
      {
        role: "system" as const,
        content: `You maintain a single-sentence summary of an ongoing conversation.
Given the current summary and a new exchange, produce an updated one-sentence summary
that preserves the most important narrative thread. Be concise but capture key facts,
decisions, and emotional tone. Focus on WHAT was discussed and any important details.

If the current summary is empty, create one from the new exchange alone.

Respond with ONLY the updated summary sentence, nothing else.`,
      },
      {
        role: "user" as const,
        content: `Current summary: ${currentSummary || "(conversation just started)"}

New exchange:
${newExchange}

Updated one-sentence summary:`,
      },
    ]);

    // Extract text from the response
    const newSummary =
      typeof response.content === "string"
        ? response.content.trim()
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("")
            .trim();

    // Store in history for future meta-summary generation
    this.summaryHistory.push(newSummary);

    return newSummary;
  }

  /**
   * Generate a "summary of summaries" to correct drift.
   *
   * This is the TWO-LEVEL mechanism from the paper. Instead of only
   * doing summary(prev + new) every turn, we periodically take the
   * buffer of recent summaries and re-derive a fresh summary.
   *
   * Why this matters:
   *   - Incremental summarisation is like the "telephone game"
   *   - Small errors compound: after 100 turns, the summary may have
   *     drifted significantly from the true conversation content
   *   - By re-summarising from the HISTORY of summaries (which were
   *     each close to accurate when created), we get a more accurate
   *     global picture
   *
   * The paper shows this eliminates cascade errors that emerge after
   * ~1,000 turns.
   *
   * @param currentSummary — the current one-sentence summary
   * @returns a fresh, consolidated summary
   */
  async generateMetaSummary(currentSummary: string): Promise<string> {
    // Need at least a few summaries to do meaningful consolidation
    if (this.summaryHistory.length < 3) return currentSummary;

    // Take the most recent summaries (cap at 10 to keep prompt small)
    const recentSummaries = this.summaryHistory.slice(-10);

    const response = await llm.invoke([
      {
        role: "system" as const,
        content: `You are given a sequence of conversation summaries taken at different points in time.
Each summary captures the state of the conversation at that moment.

Synthesize them into a single accurate one-sentence summary of the entire conversation so far.
This is a "summary of summaries" to correct for drift in incremental summarization.
Focus on the overall narrative arc and the most important facts/decisions.

Respond with ONLY the consolidated summary sentence, nothing else.`,
      },
      {
        role: "user" as const,
        content: `Summary history (oldest to newest):
${recentSummaries.map((s, i) => `${i + 1}. ${s}`).join("\n")}

Fresh consolidated summary:`,
      },
    ]);

    const metaSummary =
      typeof response.content === "string"
        ? response.content.trim()
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("")
            .trim();

    // Reset history to prevent unbounded growth
    // Keep only the meta-summary as the starting point for the next cycle
    this.summaryHistory = [metaSummary];

    return metaSummary;
  }
}
