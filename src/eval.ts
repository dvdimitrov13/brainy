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
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";

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

/** A single tool call made by the agent during retrieval */
export interface ToolCallTrace {
  tool: "recognize" | "recall";
  query: string;
  result: string;
  durationMs: number;
}

/** Tracks the agent's tool call decisions for visualization */
export interface RetrievalTrace {
  /** Ordered list of tool calls the agent made */
  toolCalls: ToolCallTrace[];
  /** Total retrieval time (all tool calls) */
  totalRetrievalMs: number;
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
  /** Two-phase retrieval trace for visualization */
  retrieval: RetrievalTrace;
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
    const sessionDate = item.haystack_dates?.[sessIdx] ?? "";

    for (let t = 0; t < sessionTurns.length; t += 2) {
      const userTurn = sessionTurns[t];
      const assistantTurn = sessionTurns[t + 1];

      if (!userTurn) continue;

      // Prepend session date so summaries and triples carry temporal context
      let exchangeText = sessionDate
        ? `[Session: ${sessionDate}]\n`
        : "";
      exchangeText += `User: ${userTurn.content}`;
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
        // Summarize each pending exchange individually (4:1 compression)
        const summaries = await compactMemory.summarizeExchanges(pendingExchanges);
        // Index each summary into HippoRAG (parallel)
        await Promise.all(summaries.map((s) => hipporag.index(s)));
        // Replace buffer with concatenated summaries
        summaryText = "[Summary of earlier conversation]\n" + summaries.join("\n\n");
        conversationBuffer = summaryText;
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

    // ── Session boundary: flush everything into HippoRAG ──
    // Each session is a separate conversation (like a new ChatGPT thread).
    // Summarize each pending exchange, then index summaries into long-term memory.
    // HippoRAG is the only memory that persists across sessions.
    if (pendingExchanges.length > 0) {
      const summaries = await compactMemory.summarizeExchanges(pendingExchanges);
      await Promise.all(summaries.map((s) => hipporag.index(s)));
      pendingExchanges = [];
    }
    conversationBuffer = "";
    hipporag.forget();
  }

  const indexingTimeMs = Date.now() - indexStart;

  // ── Agent-driven retrieval + answer generation ──
  // Same tool-calling loop as the live agent. The LLM decides whether
  // to recognize/recall based on the question and conversation context.

  const TOOLS = [
    {
      type: "function" as const,
      function: {
        name: "recognize",
        description:
          "Search long-term memory for relevant entity associations. " +
          "Returns relationship triples. Use first to check what you remember.",
        parameters: {
          type: "object" as const,
          properties: {
            query: {
              type: "string" as const,
              description: "A focused query to search memory associations.",
            },
          },
          required: ["query"],
        },
      },
    },
    {
      type: "function" as const,
      function: {
        name: "recall",
        description:
          "Retrieve full conversation summaries from long-term memory. " +
          "Use after recognize to get actual details.",
        parameters: {
          type: "object" as const,
          properties: {
            query: {
              type: "string" as const,
              description: "The query to search long-term memory for.",
            },
          },
          required: ["query"],
        },
      },
    },
  ];

  const memoryParts: string[] = [];
  if (conversationBuffer) {
    memoryParts.push(`Conversation so far:\n${conversationBuffer}`);
  }
  const memoryBlock = memoryParts.join("\n\n");

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(
      `You are a helpful, friendly assistant with long-term memory.

You have two memory tools:
1. **recognize** — search for entity associations in memory. Returns relationship triples.
2. **recall** — retrieve full conversation summaries. Use after recognize to get details.

If the user asks about something from the past, first recognize to find associations, then recall to get the details. For casual conversation, just respond directly.

Do NOT mention your memory tools or system. Just respond naturally.

${memoryBlock ? `--- Current Session ---\n${memoryBlock}\n--- End Session ---` : "(New conversation — no prior context.)"}`
    ),
    new HumanMessage(item.question),
  ];

  let lastRecognizedTriples: import("./hipporag/types.ts").Triple[] = [];
  const toolCallTraces: ToolCallTrace[] = [];
  const retrievalStart = Date.now();

  const maxToolCalls = 5;
  let generatedAnswer = "";

  for (let i = 0; i < maxToolCalls; i++) {
    const response = await llm.invoke(messages, { tools: TOOLS });

    const toolCalls = response.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      generatedAnswer =
        typeof response.content === "string"
          ? response.content
          : (response.content as Array<{ type: string; text?: string }>)
              .filter((b) => b.type === "text")
              .map((b) => b.text ?? "")
              .join("");
      break;
    }

    messages.push(response);

    for (const toolCall of toolCalls) {
      const query = (toolCall.args as { query: string }).query;
      const callStart = Date.now();

      if (toolCall.name === "recognize") {
        const triples = await hipporag.recognize(query);
        lastRecognizedTriples = triples;

        const result =
          triples.length > 0
            ? triples
                .map(
                  (t, idx) =>
                    `${idx + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
                )
                .join("\n")
            : "No relevant associations found in memory.";

        toolCallTraces.push({
          tool: "recognize",
          query,
          result,
          durationMs: Date.now() - callStart,
        });

        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? `call_${i}`,
            content: result,
          })
        );
      } else if (toolCall.name === "recall") {
        const passages = await hipporag.recall(query, lastRecognizedTriples);

        const result =
          passages.length > 0
            ? passages
                .map((p, idx) => `[Memory ${idx + 1}]: ${p.text}`)
                .join("\n\n")
            : "No relevant memories found.";

        toolCallTraces.push({
          tool: "recall",
          query,
          result,
          durationMs: Date.now() - callStart,
        });

        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? `call_${i}`,
            content: result,
          })
        );
      }
    }
  }

  const retrievalTimeMs = Date.now() - retrievalStart;

  const retrieval: RetrievalTrace = {
    toolCalls: toolCallTraces,
    totalRetrievalMs: retrievalTimeMs,
  };

  if (!generatedAnswer) {
    generatedAnswer = "I'm having trouble recalling. Could you rephrase?";
  }

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
    retrievedContext: toolCallTraces
      .filter((t) => t.tool === "recall")
      .map((t) => t.result)
      .join("\n\n"),
    conversationBuffer,
    stats,
    indexingTimeMs,
    retrievalTimeMs,
    turns,
    retrieval,
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
