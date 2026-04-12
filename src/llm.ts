/**
 * llm.ts — Singleton instances for the LLM and embedding clients.
 *
 * Python analogy:
 *   These are module-level variables. In JS/TS every file (module) is only
 *   executed once, so exporting a `const` here is the same as a Python
 *   module-level singleton.
 *
 * We expose:
 *   - `llm`          — ChatAnthropic (Claude Sonnet) for all LLM calls
 *   - `embedTexts`   — embed an array of documents via Voyage 3.5
 *   - `embedQuery`   — embed a single query via Voyage 3.5
 *
 * Rate limiting:
 *   Voyage AI's free tier is capped at 3 RPM. We add retry logic with
 *   exponential backoff so the agent gracefully handles rate limits
 *   instead of crashing. We also batch all embeddings into as few
 *   API calls as possible.
 */

import { ChatAnthropic } from "@langchain/anthropic";
import { VoyageAIClient } from "voyageai";

// ──────────────────────────────────────────────
// LLMs — two tiers for different tasks
// ──────────────────────────────────────────────
// Reads ANTHROPIC_API_KEY from process.env automatically (Bun loads .env).

/**
 * Primary LLM (Claude Sonnet) — used for response generation and the
 * LLM judge in evaluation. Higher quality, slower.
 */
export const llm = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  temperature: 0,
  maxTokens: 2048,
});

/**
 * Fast LLM (Claude Haiku) — used for high-volume "processing" tasks:
 * triple extraction (OpenIE) and compact memory summarisation.
 *
 * These tasks are structured (extract JSON, produce a summary sentence)
 * and Haiku handles them well at ~5x the speed.
 */
export const llmFast = new ChatAnthropic({
  model: "claude-haiku-4-5-20251001",
  temperature: 0,
  maxTokens: 1024,
});

/**
 * Mid-tier LLM (Claude Sonnet) — used for quality-sensitive processing:
 * recognition memory filtering (triple relevance judgement).
 *
 * Recognition memory is the gate that decides which triples seed PPR,
 * so filtering quality directly impacts retrieval accuracy. Sonnet's
 * stronger reasoning helps here more than in mechanical extraction.
 */
export const llmMid = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  temperature: 0,
  maxTokens: 1024,
});

// ──────────────────────────────────────────────
// Embeddings — Voyage AI 3.5
// ──────────────────────────────────────────────
// Reads VOYAGE_API_KEY from process.env automatically.
const voyageClient = new VoyageAIClient();

/**
 * Sleep for a given number of milliseconds.
 *
 * Python equivalent: `await asyncio.sleep(seconds)`
 *
 * TS note: `new Promise<void>(resolve => setTimeout(resolve, ms))` creates
 * a promise that resolves after `ms` milliseconds. `setTimeout` is like
 * Python's `loop.call_later()`.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Call a Voyage AI embed function with retry + exponential backoff.
 *
 * On 429 (rate limit), waits and retries up to `maxRetries` times.
 * Backoff schedule: 20s, 40s, 60s (generous because free tier is 3 RPM).
 *
 * @param fn — the async function to call
 * @param maxRetries — maximum number of retries (default 3)
 * @returns the result of the function
 */
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: unknown) {
      // Check if it's a rate limit error (429)
      const isRateLimit =
        error instanceof Error &&
        (error.message.includes("429") ||
          error.message.includes("Too Many Requests") ||
          (error as { statusCode?: number }).statusCode === 429);

      if (isRateLimit && attempt < maxRetries) {
        const waitMs = (attempt + 1) * 20_000; // 20s, 40s, 60s
        console.log(
          `  [Embedding] Rate limited, waiting ${waitMs / 1000}s (attempt ${attempt + 1}/${maxRetries})...`
        );
        await sleep(waitMs);
        continue;
      }

      throw error; // Not a rate limit, or retries exhausted
    }
  }

  // TypeScript requires this, but it's unreachable
  throw new Error("Retry logic failed unexpectedly");
}

/**
 * Embed one or more **documents** (passages, entity names, triples).
 *
 * Voyage AI distinguishes `inputType: "document"` vs `"query"` — using the
 * correct type improves retrieval quality because the model applies slightly
 * different projections for asymmetric search.
 *
 * Includes retry logic for rate limiting (free tier: 3 RPM).
 *
 * @param texts  — array of strings to embed
 * @returns        array of number arrays (one embedding per input text)
 */
export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];

  return withRetry(async () => {
    const response = await voyageClient.embed({
      input: texts,
      model: "voyage-3.5",
      inputType: "document",
    });

    // response.data is an array of { embedding: number[], index: number }
    return (response.data ?? []).map((item) => item.embedding ?? []);
  });
}

/**
 * Embed a single **query** string (the user's question / retrieval query).
 *
 * Uses `inputType: "query"` so the model optimises the embedding for
 * retrieving relevant documents (asymmetric search).
 *
 * Includes retry logic for rate limiting.
 *
 * @param text  — the query string
 * @returns       a single embedding vector
 */
export async function embedQuery(text: string): Promise<number[]> {
  return withRetry(async () => {
    const response = await voyageClient.embed({
      input: text,
      model: "voyage-3.5",
      inputType: "query",
    });

    return response.data?.[0]?.embedding ?? [];
  });
}
