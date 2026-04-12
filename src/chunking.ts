/**
 * chunking.ts — Chunk, rerank, and pack retrieved passages.
 *
 * When the recall tool returns passages, we don't want to dump full
 * exchanges into the LLM context. Instead:
 *
 *   1. Split each exchange into ~256-token chunks with 32-token overlap
 *   2. Respect user/assistant boundary (no chunk mixes both)
 *   3. Rerank all chunks across all passages via Voyage rerank-2
 *   4. Pack chunks in reranked order until we hit a token budget (1024)
 *
 * This ensures the agent gets the most relevant snippets from across
 * all retrieved passages, tightly packed within a bounded context.
 *
 * TS note for Python devs:
 *   Recursive character splitting is like LangChain's
 *   RecursiveCharacterTextSplitter — try paragraph, line, sentence,
 *   word boundaries in order, falling back to raw character split.
 */

import { VoyageAIClient } from "voyageai";

const voyageClient = new VoyageAIClient();

/** ~256 tokens in characters (4 chars/token heuristic) */
const CHUNK_SIZE = 1024;
/** ~32 tokens overlap in characters */
const CHUNK_OVERLAP = 128;
/** Token budget for the packed result (~1024 tokens) */
const TOKEN_BUDGET = 1024;

/** Separators for recursive splitting, tried in order */
const SEPARATORS = ["\n\n", "\n", ". ", ", ", " "];

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Split text recursively using progressively finer separators.
 *
 * Tries paragraph breaks first, then line breaks, sentences, etc.
 * Each chunk is ≤ CHUNK_SIZE chars with CHUNK_OVERLAP overlap.
 */
function recursiveSplit(
  text: string,
  separators: string[] = SEPARATORS
): string[] {
  if (text.length <= CHUNK_SIZE) return [text];

  // Find the first separator that actually splits the text
  const sep = separators.find((s) => text.includes(s));

  if (!sep) {
    // No separator works — hard split by character with overlap
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      chunks.push(text.slice(start, start + CHUNK_SIZE));
      start += CHUNK_SIZE - CHUNK_OVERLAP;
    }
    return chunks;
  }

  // Split on this separator
  const parts = text.split(sep);
  const chunks: string[] = [];
  let current = "";

  for (const part of parts) {
    const candidate = current ? current + sep + part : part;

    if (candidate.length > CHUNK_SIZE) {
      if (current) {
        chunks.push(current);
      }
      // If a single part is too long, recurse with finer separators
      if (part.length > CHUNK_SIZE) {
        const subSeps = separators.slice(separators.indexOf(sep) + 1);
        chunks.push(...recursiveSplit(part, subSeps));
        current = "";
      } else {
        current = part;
      }
    } else {
      current = candidate;
    }
  }

  if (current) chunks.push(current);

  // Add overlap between adjacent chunks
  if (chunks.length > 1 && CHUNK_OVERLAP > 0) {
    const overlapped: string[] = [chunks[0]!];
    for (let i = 1; i < chunks.length; i++) {
      const prev = chunks[i - 1]!;
      const overlapText = prev.slice(-CHUNK_OVERLAP);
      overlapped.push(overlapText + chunks[i]!);
    }
    return overlapped;
  }

  return chunks;
}

/**
 * Split an exchange into chunks, respecting user/assistant boundary.
 *
 * First separates the user and assistant parts, then chunks each
 * independently. No chunk will contain both user and assistant content.
 *
 * @param exchange — "User: ...\nAssistant: ..." text
 * @returns array of chunks, each tagged with role
 */
export function chunkExchange(
  exchange: string
): { text: string; role: "user" | "assistant" }[] {
  // Split on the "Assistant:" boundary
  const assistantIdx = exchange.indexOf("\nAssistant:");

  let userPart: string;
  let assistantPart: string | null;

  if (assistantIdx >= 0) {
    userPart = exchange.slice(0, assistantIdx);
    assistantPart = exchange.slice(assistantIdx + 1); // skip the \n
  } else {
    userPart = exchange;
    assistantPart = null;
  }

  const chunks: { text: string; role: "user" | "assistant" }[] = [];

  // Chunk user part
  for (const chunk of recursiveSplit(userPart)) {
    if (chunk.trim()) {
      chunks.push({ text: chunk.trim(), role: "user" });
    }
  }

  // Chunk assistant part
  if (assistantPart) {
    for (const chunk of recursiveSplit(assistantPart)) {
      if (chunk.trim()) {
        chunks.push({ text: chunk.trim(), role: "assistant" });
      }
    }
  }

  return chunks;
}

/**
 * Chunk, rerank, and pack passages within a token budget.
 *
 * This is the main function used by both the recall tool and the eval.
 *
 * @param query — the retrieval query
 * @param passages — raw passage texts from HippoRAG retrieval
 * @param tokenBudget — max tokens to return (default 1024)
 * @returns packed string of the most relevant chunks within budget
 */
export async function chunkRerankPack(
  query: string,
  passages: string[],
  tokenBudget: number = TOKEN_BUDGET
): Promise<string> {
  if (passages.length === 0) return "No relevant memories found.";

  // ── Step 1: Chunk all passages ──
  const allChunks: { text: string; role: string; passageIdx: number }[] = [];

  for (let pIdx = 0; pIdx < passages.length; pIdx++) {
    const chunks = chunkExchange(passages[pIdx]!);
    for (const chunk of chunks) {
      allChunks.push({
        text: chunk.text,
        role: chunk.role,
        passageIdx: pIdx,
      });
    }
  }

  if (allChunks.length === 0) return "No relevant memories found.";

  // If everything fits in budget, skip reranking
  const totalTokens = allChunks.reduce(
    (sum, c) => sum + estimateTokens(c.text),
    0
  );

  if (totalTokens <= tokenBudget) {
    return allChunks.map((c) => c.text).join("\n\n");
  }

  // ── Step 2: Rerank all chunks ──
  const reranked = await voyageClient.rerank({
    query,
    documents: allChunks.map((c) => c.text),
    model: "rerank-2",
  });

  // ── Step 3: Pack in reranked order until budget is exhausted ──
  const packed: string[] = [];
  let usedTokens = 0;

  for (const item of reranked.data ?? []) {
    const chunk = allChunks[item.index!];
    if (!chunk) continue;

    const chunkTokens = estimateTokens(chunk.text);
    if (usedTokens + chunkTokens > tokenBudget) break;

    packed.push(chunk.text);
    usedTokens += chunkTokens;
  }

  if (packed.length === 0 && allChunks.length > 0) {
    // At least return the top-ranked chunk even if it exceeds budget
    const topIdx = reranked.data?.[0]?.index;
    if (topIdx !== undefined && allChunks[topIdx]) {
      packed.push(allChunks[topIdx].text.slice(0, tokenBudget * 4));
    }
  }

  return packed.join("\n\n");
}
