/**
 * compact-memory.ts — Pressure-based conversation buffer with summarization.
 *
 * Manages the conversation buffer and provides two levels of summarization:
 *
 *   1. Per-exchange summaries — each exchange gets compressed to ~25% of its
 *      original size (4:1 ratio). These dense summaries are what HippoRAG
 *      indexes, preserving a 1:1 link between exchanges and passages.
 *
 *   2. Buffer summarization — when the buffer exceeds the token threshold,
 *      all pending exchanges are individually summarized and the summaries
 *      replace the buffer content.
 *
 * The summaries use high information density: telegraphic style, no
 * pleasantries, focused on facts, names, numbers, decisions, preferences.
 *
 * TS note for Python devs:
 *   - `Promise.all(items.map(fn))` is like `asyncio.gather(*[fn(i) for i in items])`
 */

import { llmFast } from "../llm.ts";

/**
 * Rough token count estimate (~4 chars per token).
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Token threshold before summarization kicks in */
const TOKEN_THRESHOLD = 1024;

export class CompactMemory {
  /**
   * Check if the conversation buffer has exceeded memory pressure.
   */
  shouldSummarize(buffer: string): boolean {
    return estimateTokens(buffer) > TOKEN_THRESHOLD;
  }

  /**
   * Append a new exchange to the conversation buffer.
   */
  append(currentBuffer: string, newExchange: string): string {
    if (!currentBuffer) return newExchange;
    return `${currentBuffer}\n\n${newExchange}`;
  }

  /**
   * Summarize a single exchange at ~4:1 compression ratio.
   *
   * Produces a dense, telegraphic summary preserving key facts.
   * This summary is what gets indexed in HippoRAG, maintaining
   * a 1:1 link between exchanges and knowledge graph passages.
   *
   * @param exchange — a single "User: ...\nAssistant: ..." exchange
   * @returns compressed summary (~25% of original size)
   */
  async summarizeExchange(exchange: string): Promise<string> {
    const targetTokens = Math.max(30, Math.round(estimateTokens(exchange) / 4));

    const response = await llmFast.invoke([
      {
        role: "system" as const,
        content: `Compress this conversation exchange to ~${targetTokens} tokens.
Use high information density: telegraphic style, skip pleasantries and filler.
Preserve: names, numbers, dates, facts, decisions, preferences, key details.
Drop: greetings, acknowledgements, politeness, repetition, elaboration.
Write as dense notes, not full sentences. Keep the User/Assistant structure.

Respond with ONLY the compressed exchange, nothing else.`,
      },
      {
        role: "user" as const,
        content: exchange,
      },
    ]);

    const summary =
      typeof response.content === "string"
        ? response.content.trim()
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("")
            .trim();

    return summary;
  }

  /**
   * Summarize all pending exchanges individually (in parallel).
   *
   * Returns an array of dense summaries, one per exchange. These
   * are used both for HippoRAG indexing and as the new buffer content.
   *
   * @param exchanges — array of raw exchange texts
   * @returns array of compressed summaries (same order)
   */
  async summarizeExchanges(exchanges: string[]): Promise<string[]> {
    return Promise.all(exchanges.map((ex) => this.summarizeExchange(ex)));
  }

  /**
   * Create a summary-of-summaries for the conversation buffer.
   *
   * Takes the individual exchange summaries and compresses them into
   * a single narrative thread. This is what replaces the buffer —
   * the individual summaries are already indexed in HippoRAG, so
   * the buffer just needs the high-level context.
   *
   * @param summaries — array of per-exchange summaries
   * @returns a single compressed narrative
   */
  async summarizeSummaries(summaries: string[]): Promise<string> {
    const joined = summaries.join("\n\n");

    const response = await llmFast.invoke([
      {
        role: "system" as const,
        content: `Compress these conversation summaries into a single brief narrative (2-3 sentences).
Capture the overall thread: what topics were discussed, key facts established,
and where the conversation is heading. Details are stored separately in
long-term memory — this just needs to provide context for what comes next.

Respond with ONLY the narrative, nothing else.`,
      },
      {
        role: "user" as const,
        content: joined,
      },
    ]);

    const summary =
      typeof response.content === "string"
        ? response.content.trim()
        : (response.content as Array<{ type: string; text?: string }>)
            .filter((block) => block.type === "text")
            .map((block) => block.text ?? "")
            .join("")
            .trim();

    return `[Summary of earlier conversation]\n${summary}`;
  }
}
