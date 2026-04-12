/**
 * eval.ts — LongMemEval benchmark evaluation logic.
 *
 * Exports core evaluation functions that can be used by both the CLI
 * and the API server. The CLI entrypoint is at the bottom (runs only
 * when this file is executed directly, not when imported).
 *
 * Evaluates Brainy's dual-memory system against the LongMemEval benchmark
 * (ICLR 2025). The benchmark tests 5 long-term memory abilities:
 *   1. Information Extraction (single-session-user/assistant/preference)
 *   2. Multi-Session Reasoning
 *   3. Temporal Reasoning
 *   4. Knowledge Updates
 *
 * Usage (CLI):
 *   bun run src/eval.ts --count 2
 *   bun run src/eval.ts --count all
 *   bun run src/eval.ts --type multi-session --count 5
 */

import { HippoRAG } from "./hipporag/index.ts";
import { CompactMemory } from "./memory/compact-memory.ts";
import { llm } from "./llm.ts";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ══════════════════════════════════════════════
// TYPES (exported for server + frontend)
// ══════════════════════════════════════════════

/** A single turn in a conversation session */
export interface Turn {
  role: "user" | "assistant";
  content: string;
  has_answer?: boolean;
}

/** A single evaluation item from the LongMemEval dataset */
export interface EvalItem {
  question_id: string;
  question_type: string;
  question: string;
  answer: string;
  question_date: string;
  haystack_dates: string[];
  haystack_session_ids: string[];
  haystack_sessions: Record<string, Turn>[];
  answer_session_ids: string[];
}

/** Snapshot of a single indexing turn — for frontend visualization */
export interface TurnSnapshot {
  turnIndex: number;
  sessionIndex: number;
  exchangeText: string;
  bufferTokensBefore: number;
  /** Token count after appending the exchange but before summarization */
  bufferTokensPeak: number;
  bufferTokensAfter: number;
  summarized: boolean;
  summaryText?: string;
  kgStats: { passages: number; entities: number; facts: number };
}

/** Result of evaluating a single question — enriched with turn-level data */
export interface EvalResult {
  questionId: string;
  questionType: string;
  question: string;
  expectedAnswer: string;
  generatedAnswer: string;
  correct: boolean;
  retrievedContext: string;
  conversationBuffer: string;
  stats: { passages: number; entities: number; facts: number };
  indexingTimeMs: number;
  retrievalTimeMs: number;
  turns: TurnSnapshot[];
}

// ══════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ══════════════════════════════════════════════
// LLM JUDGE
// ══════════════════════════════════════════════

async function judgeAnswer(
  question: string,
  expectedAnswer: string,
  generatedAnswer: string
): Promise<boolean> {
  const response = await llm.invoke([
    new SystemMessage(
      `You are an evaluation judge. Given a question, an expected answer, and a generated answer, determine if the generated answer is correct.

The generated answer is CORRECT if:
- It contains the key information from the expected answer (semantic match, not exact string)
- It may contain additional details beyond the expected answer — that's fine
- It may use different wording — what matters is the factual content

The generated answer is INCORRECT if:
- It misses the key facts from the expected answer
- It contradicts the expected answer
- It says "I don't know" or similar when an answer exists
- It provides a fundamentally different answer

Respond with ONLY "CORRECT" or "INCORRECT", nothing else.`
    ),
    new HumanMessage(
      `Question: ${question}

Expected answer: ${expectedAnswer}

Generated answer: ${generatedAnswer}

Verdict:`
    ),
  ]);

  const verdict =
    typeof response.content === "string"
      ? response.content.trim().toUpperCase()
      : "";

  return verdict.includes("CORRECT") && !verdict.includes("INCORRECT");
}

// ══════════════════════════════════════════════
// CORE EVALUATION LOGIC (exported)
// ══════════════════════════════════════════════

/**
 * Load the LongMemEval dataset from disk.
 */
export async function loadDataset(
  dataset: string = "oracle"
): Promise<EvalItem[]> {
  const dataPath =
    dataset === "s"
      ? "data/longmemeval_s_cleaned.json"
      : "data/longmemeval_oracle.json";

  return await Bun.file(dataPath).json();
}

/**
 * Select a subset of items for evaluation.
 *
 * If type is specified, filters to that type. If count is a number,
 * takes N per type for balanced evaluation.
 */
export function selectItems(
  allItems: EvalItem[],
  count: number | "all",
  type?: string
): EvalItem[] {
  let items = type
    ? allItems.filter((i) => i.question_type === type)
    : allItems;

  if (count !== "all") {
    if (type) {
      items = items.slice(0, count);
    } else {
      const byType = new Map<string, EvalItem[]>();
      for (const item of items) {
        if (!byType.has(item.question_type)) {
          byType.set(item.question_type, []);
        }
        byType.get(item.question_type)!.push(item);
      }

      items = [];
      for (const [, typeItems] of byType) {
        items.push(...typeItems.slice(0, count));
      }
    }
  }

  return items;
}

/**
 * Count total turn pairs in an eval item (for progress reporting).
 */
export function countTurnPairs(item: EvalItem): number {
  let count = 0;
  for (const session of item.haystack_sessions) {
    const turns = Object.values(session) as Turn[];
    count += Math.ceil(turns.length / 2);
  }
  return count;
}

/**
 * Callback fired after each turn during evaluation.
 * Used by the SSE server to stream turn-level progress.
 */
export type OnTurnCallback = (snapshot: TurnSnapshot, totalTurns: number) => void;

/**
 * Evaluate a single LongMemEval question.
 * Captures turn-level snapshots for visualization.
 *
 * @param item — the question to evaluate
 * @param onTurn — optional callback fired after each turn completes
 */
export async function evaluateQuestion(
  item: EvalItem,
  onTurn?: OnTurnCallback
): Promise<EvalResult> {
  const hipporag = new HippoRAG();
  const compactMemory = new CompactMemory();
  let conversationBuffer = "";
  let pendingExchanges: string[] = [];
  const turns: TurnSnapshot[] = [];
  let globalTurnIndex = 0;

  const totalTurns = countTurnPairs(item);
  const indexStart = Date.now();

  for (let sessIdx = 0; sessIdx < item.haystack_sessions.length; sessIdx++) {
    const session = item.haystack_sessions[sessIdx]!;
    const sessionTurns = Object.values(session) as Turn[];

    for (let t = 0; t < sessionTurns.length; t += 2) {
      const userTurn = sessionTurns[t];
      const assistantTurn = sessionTurns[t + 1];

      if (!userTurn) continue;

      let exchangeText = `User: ${userTurn.content}`;
      if (assistantTurn) {
        exchangeText += `\nAssistant: ${assistantTurn.content}`;
      }

      const bufferTokensBefore = estimateTokens(conversationBuffer);

      conversationBuffer = compactMemory.append(
        conversationBuffer,
        exchangeText
      );
      pendingExchanges.push(exchangeText);

      // Capture peak token count AFTER append but BEFORE summarization
      const bufferTokensPeak = estimateTokens(conversationBuffer);

      let summarized = false;
      let summaryText: string | undefined;

      if (compactMemory.shouldSummarize(conversationBuffer)) {
        // Index each pending exchange as a separate passage (in parallel)
        // so HippoRAG gets granular passages with focused triples
        const [summary] = await Promise.all([
          compactMemory.summarize(conversationBuffer),
          ...pendingExchanges.map((ex) => hipporag.index(ex)),
        ]);
        summaryText = summary;
        conversationBuffer = summary;
        pendingExchanges = [];
        summarized = true;
        hipporag.forget();
      }

      const bufferTokensAfter = estimateTokens(conversationBuffer);
      const kgStats = hipporag.getStats();

      const snapshot: TurnSnapshot = {
        turnIndex: globalTurnIndex++,
        sessionIndex: sessIdx,
        exchangeText,
        bufferTokensBefore,
        bufferTokensPeak,
        bufferTokensAfter,
        summarized,
        summaryText,
        kgStats: {
          passages: kgStats.passages,
          entities: kgStats.entities,
          facts: kgStats.facts,
        },
      };

      turns.push(snapshot);
      onTurn?.(snapshot, totalTurns);
    }
  }

  // Index any remaining pending exchanges that didn't trigger pressure
  if (pendingExchanges.length > 0) {
    await Promise.all(pendingExchanges.map((ex) => hipporag.index(ex)));
  }

  const indexingTimeMs = Date.now() - indexStart;

  const retrievalStart = Date.now();
  const passages = await hipporag.retrieve(item.question, 5);
  const retrievalTimeMs = Date.now() - retrievalStart;

  const retrievedContext = passages
    .map((p, i) => `[Memory ${i + 1}]: ${p.text}`)
    .join("\n\n");

  const memoryBlock = [
    conversationBuffer && `Conversation context:\n${conversationBuffer}`,
    retrievedContext && `Relevant long-term memories:\n${retrievedContext}`,
  ]
    .filter(Boolean)
    .join("\n\n");

  const response = await llm.invoke([
    new SystemMessage(
      `You are a helpful assistant with long-term memory of past conversations.
Use the memories provided below to answer the user's question accurately.
If you can find the answer in your memories, provide it directly and concisely.
If you truly cannot find the answer, say so.

--- Your Memories ---
${memoryBlock}
--- End Memories ---`
    ),
    new HumanMessage(item.question),
  ]);

  const generatedAnswer =
    typeof response.content === "string"
      ? response.content
      : (response.content as Array<{ type: string; text?: string }>)
          .filter((b) => b.type === "text")
          .map((b) => b.text ?? "")
          .join("");

  const correct = await judgeAnswer(
    item.question,
    item.answer,
    generatedAnswer
  );

  const stats = hipporag.getStats();

  return {
    questionId: item.question_id,
    questionType: item.question_type,
    question: item.question,
    expectedAnswer: item.answer,
    generatedAnswer,
    correct,
    retrievedContext,
    conversationBuffer,
    stats,
    indexingTimeMs,
    retrievalTimeMs,
    turns,
  };
}

// ══════════════════════════════════════════════
// CLI ENTRYPOINT
// Only runs when executed directly (not imported)
// ══════════════════════════════════════════════

function parseArgs(): { count: number | "all"; type?: string; dataset: string } {
  const args = process.argv.slice(2);
  let count: number | "all" = 5;
  let type: string | undefined;
  let dataset = "oracle";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--count") {
      const val = args[i + 1];
      count = val === "all" ? "all" : parseInt(val ?? "5", 10);
      i++;
    } else if (args[i] === "--type") {
      type = args[i + 1];
      i++;
    } else if (args[i] === "--dataset") {
      dataset = args[i + 1] ?? "oracle";
      i++;
    }
  }

  return { count, type, dataset };
}

async function main() {
  const { count, type, dataset } = parseArgs();

  console.log("Loading dataset...");
  const allItems = await loadDataset(dataset);
  console.log(`Loaded ${allItems.length} questions.\n`);

  const items = selectItems(allItems, count, type);
  console.log(`Running evaluation on ${items.length} questions...\n`);

  const results: EvalResult[] = [];
  let completed = 0;

  for (const item of items) {
    completed++;
    const prefix = `[${completed}/${items.length}]`;

    try {
      process.stdout.write(
        `${prefix} ${item.question_type.padEnd(28)} "${item.question.slice(0, 50)}..." `
      );

      const result = await evaluateQuestion(item);
      results.push(result);

      const mark = result.correct ? "PASS" : "FAIL";
      const kg = result.stats;
      console.log(
        `${mark} (${kg.entities}e/${kg.facts}f, idx:${(result.indexingTimeMs / 1000).toFixed(0)}s, ret:${(result.retrievalTimeMs / 1000).toFixed(0)}s)`
      );

      if (!result.correct) {
        console.log(`       Expected: ${result.expectedAnswer}`);
        console.log(
          `       Got:      ${result.generatedAnswer.slice(0, 120)}...`
        );
      }
    } catch (error) {
      console.log(`ERROR: ${error}`);
    }
  }

  // Report
  console.log("\n" + "=".repeat(70));
  const totalCorrect = results.filter((r) => r.correct).length;
  console.log(
    `  Overall: ${totalCorrect}/${results.length} (${((totalCorrect / results.length) * 100).toFixed(1)}%)`
  );
  console.log("=".repeat(70) + "\n");

  const outputPath = `data/eval_results_${dataset}_${new Date().toISOString().slice(0, 10)}.json`;
  await Bun.write(outputPath, JSON.stringify(results, null, 2));
  console.log(`Results saved to ${outputPath}`);
}

// Only run CLI if this file is the entrypoint (not imported by server)
const isMainModule = import.meta.path === Bun.main;
if (isMainModule) {
  main().catch((err) => {
    console.error("Evaluation failed:", err);
    process.exit(1);
  });
}
