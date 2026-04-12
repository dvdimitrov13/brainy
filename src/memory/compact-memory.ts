/**
 * compact-memory.ts — Pressure-based conversation buffer with summarization.
 *
 * Instead of summarizing every turn (lossy from turn 1), this module
 * accumulates real conversation turns and only compresses when the
 * buffer exceeds a token threshold. This preserves full fidelity for
 * recent exchanges while keeping memory bounded.
 *
 * The flow:
 *   1. Each turn, `append()` adds the new exchange to the buffer
 *   2. `shouldSummarize()` checks if the buffer exceeds the token limit
 *   3. If yes, `summarize()` compresses the entire buffer into a
 *      compact paragraph and returns it as the new buffer contents
 *   4. At summarization time, the caller also indexes the summary
 *      into HippoRAG for long-term retrieval
 *
 * Brain analogy: This is like short-term / working memory. You hold
 * recent events in full detail, but older events get compressed into
 * gist as new information competes for the same limited capacity.
 *
 * TS note for Python devs:
 *   - `private` fields like `TOKEN_THRESHOLD` are class-level constants
 *     that can't be accessed outside the class (Python uses _ prefix).
 *   - Template literals (`${var}`) are like Python f-strings.
 */

import { llmFast } from "../llm.ts";

/**
 * Rough token count estimate.
 *
 * LLMs use subword tokenizers (BPE), so there's no exact char-to-token
 * mapping. The common heuristic is ~4 characters per token for English.
 * We use this for the memory pressure check — it doesn't need to be
 * exact, just close enough to trigger summarization at the right time.
 *
 * @param text — the text to estimate tokens for
 * @returns approximate token count
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Token threshold before summarization kicks in */
const TOKEN_THRESHOLD = 1024;

export class CompactMemory {
  /**
   * Check if the conversation buffer has exceeded memory pressure.
   *
   * @param buffer — the current conversation buffer text
   * @returns true if the buffer should be summarized
   */
  shouldSummarize(buffer: string): boolean {
    return estimateTokens(buffer) > TOKEN_THRESHOLD;
  }

  /**
   * Append a new exchange to the conversation buffer.
   *
   * Simply concatenates the new exchange onto the existing buffer
   * with a separator. No LLM call needed — this is just accumulation.
   *
   * @param currentBuffer — the existing buffer content
   * @param newExchange   — the new "User: ...\nAssistant: ..." text
   * @returns the updated buffer with the new exchange appended
   */
  append(currentBuffer: string, newExchange: string): string {
    if (!currentBuffer) return newExchange;
    return `${currentBuffer}\n\n${newExchange}`;
  }

  /**
   * Summarize the conversation buffer under memory pressure.
   *
   * Compresses the entire buffer into a compact paragraph that
   * preserves the key facts, decisions, preferences, and narrative
   * arc. This is only called when the buffer exceeds TOKEN_THRESHOLD.
   *
   * Unlike the old approach (summarize every turn into one sentence),
   * this produces a richer summary because it has access to the full
   * multi-turn context at compression time.
   *
   * @param buffer — the full conversation buffer to compress
   * @returns a compact summary paragraph
   */
  async summarize(buffer: string): Promise<string> {
    const response = await llmFast.invoke([
      {
        role: "system" as const,
        content: `You compress a conversation buffer into a compact summary paragraph.
Preserve ALL important information: facts, names, preferences, decisions, plans,
emotional tone, and the narrative thread. Be concise but thorough — this summary
replaces the original text, so anything you drop is lost.

Focus on WHAT was discussed, WHO was mentioned, and any specific details
(numbers, dates, names, preferences) that would be important to recall later.

Respond with ONLY the summary paragraph, nothing else.`,
      },
      {
        role: "user" as const,
        content: `Conversation buffer to compress:\n\n${buffer}`,
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
