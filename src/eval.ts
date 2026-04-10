/**
 * eval.ts — LongMemEval benchmark evaluation harness.
 *
 * Evaluates Brainy's dual-memory system against the LongMemEval benchmark
 * (ICLR 2025). The benchmark tests 5 long-term memory abilities:
 *
 *   1. Information Extraction (single-session-user, single-session-assistant,
 *      single-session-preference) — recall specific facts from conversation
 *   2. Multi-Session Reasoning — synthesize info across separate sessions
 *   3. Temporal Reasoning — understand time-based relationships
 *   4. Knowledge Updates — handle evolving/contradicting information
 *
 * For each question:
 *   1. Reset memory to fresh state
 *   2. Feed all haystack sessions into HippoRAG + Compact Memory (indexing)
 *   3. Run retrieval + generation for the question
 *   4. Use LLM judge (Claude) to evaluate if the answer is correct
 *   5. Record scores by category
 *
 * Usage:
 *   bun run src/eval.ts                    # run default sample (5 per category)
 *   bun run src/eval.ts --count 10         # 10 per category
 *   bun run src/eval.ts --count all        # run all 500
 *   bun run src/eval.ts --type multi-session --count 20
 *
 * TS note for Python devs:
 *   `process.argv` is like `sys.argv` — an array of command-line arguments.
 *   `Bun.file(path).json()` reads and parses a JSON file (like json.load(open(path))).
 */

import { HippoRAG } from "./hipporag/index.ts";
import { CompactMemory } from "./memory/compact-memory.ts";
import { llm, embedQuery } from "./llm.ts";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ══════════════════════════════════════════════
// TYPES
// ══════════════════════════════════════════════

/** A single turn in a conversation session */
interface Turn {
  role: "user" | "assistant";
  content: string;
  has_answer?: boolean;
}

/** A single evaluation item from the LongMemEval dataset */
interface EvalItem {
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

/** Result of evaluating a single question */
interface EvalResult {
  questionId: string;
  questionType: string;
  question: string;
  expectedAnswer: string;
  generatedAnswer: string;
  correct: boolean;
  retrievedContext: string;
  compactSummary: string;
  stats: { passages: number; entities: number; facts: number };
  indexingTimeMs: number;
  retrievalTimeMs: number;
}

// ══════════════════════════════════════════════
// CLI ARGUMENT PARSING
// ══════════════════════════════════════════════

function parseArgs(): { count: number | "all"; type?: string; dataset: string } {
  const args = process.argv.slice(2);
  let count: number | "all" = 5; // default: 5 per category
  let type: string | undefined;
  let dataset = "oracle"; // "oracle" or "s"

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

// ══════════════════════════════════════════════
// LLM JUDGE
// ══════════════════════════════════════════════

/**
 * Use Claude as a judge to evaluate if the generated answer is correct.
 *
 * This mirrors LongMemEval's evaluation approach (they use GPT-4o).
 * The judge checks semantic equivalence, not exact string match —
 * "GPS wasn't working" and "GPS system not functioning correctly"
 * should both count as correct.
 *
 * @returns true if the answer is judged correct
 */
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
// CORE EVALUATION LOGIC
// ══════════════════════════════════════════════

/**
 * Evaluate a single LongMemEval question.
 *
 * Steps:
 *   1. Create fresh HippoRAG + CompactMemory instances
 *   2. Feed all haystack sessions (user+assistant pairs) into memory
 *   3. Retrieve relevant context for the question
 *   4. Generate an answer using LLM + retrieved context
 *   5. Judge correctness via LLM
 */
async function evaluateQuestion(item: EvalItem): Promise<EvalResult> {
  // ── Step 1: Fresh memory instances ──
  const hipporag = new HippoRAG();
  const compactMemory = new CompactMemory();
  let compactSummary = "";

  // ── Step 2: Index all sessions ──
  const indexStart = Date.now();

  for (let sessIdx = 0; sessIdx < item.haystack_sessions.length; sessIdx++) {
    const session = item.haystack_sessions[sessIdx]!;
    // Sessions are objects with numeric string keys, not arrays
    const turns = Object.values(session) as Turn[];

    // Pair up user/assistant turns into exchanges and index them
    for (let t = 0; t < turns.length; t += 2) {
      const userTurn = turns[t];
      const assistantTurn = turns[t + 1];

      if (!userTurn) continue;

      // Build exchange text (user + assistant if available)
      let exchangeText = `User: ${userTurn.content}`;
      if (assistantTurn) {
        exchangeText += `\nAssistant: ${assistantTurn.content}`;
      }

      // Index in HippoRAG (extract triples, embed, add to KG)
      // and update compact memory in parallel
      const [newSummary] = await Promise.all([
        compactMemory.update(compactSummary, exchangeText),
        hipporag.index(exchangeText),
      ]);

      compactSummary = newSummary;
    }
  }

  const indexingTimeMs = Date.now() - indexStart;

  // ── Step 3: Retrieve relevant context for the question ──
  const retrievalStart = Date.now();
  const passages = await hipporag.retrieve(item.question, 5);
  const retrievalTimeMs = Date.now() - retrievalStart;

  const retrievedContext = passages
    .map((p, i) => `[Memory ${i + 1}]: ${p.text}`)
    .join("\n\n");

  // ── Step 4: Generate answer ──
  const memoryBlock = [
    compactSummary && `Conversation summary: ${compactSummary}`,
    retrievedContext && `Relevant memories:\n${retrievedContext}`,
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

  // ── Step 5: Judge correctness ──
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
    compactSummary,
    stats,
    indexingTimeMs,
    retrievalTimeMs,
  };
}

// ══════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════

async function main() {
  const { count, type, dataset } = parseArgs();

  // Load dataset
  const dataPath =
    dataset === "s"
      ? "data/longmemeval_s_cleaned.json"
      : "data/longmemeval_oracle.json";

  console.log(`Loading ${dataPath}...`);
  const allItems: EvalItem[] = await Bun.file(dataPath).json();
  console.log(`Loaded ${allItems.length} questions.\n`);

  // Filter by type if specified
  let items = type
    ? allItems.filter((i) => i.question_type === type)
    : allItems;

  if (type) {
    console.log(`Filtered to ${items.length} questions of type "${type}".`);
  }

  // Sample if count is not "all"
  if (count !== "all") {
    if (type) {
      // Single type: just take the first N
      items = items.slice(0, count);
    } else {
      // Multiple types: take N per type for balanced evaluation
      const byType = new Map<string, EvalItem[]>();
      for (const item of items) {
        if (!byType.has(item.question_type)) {
          byType.set(item.question_type, []);
        }
        byType.get(item.question_type)!.push(item);
      }

      items = [];
      for (const [typeName, typeItems] of byType) {
        items.push(...typeItems.slice(0, count));
        console.log(
          `  ${typeName}: ${Math.min(count, typeItems.length)} questions`
        );
      }
    }
  }

  console.log(`\nRunning evaluation on ${items.length} questions...\n`);

  // ── Run evaluation ──
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

  // ══════════════════════════════════════════════
  // RESULTS REPORT
  // ══════════════════════════════════════════════

  console.log("\n" + "═".repeat(70));
  console.log("  LONGMEMEVAL EVALUATION RESULTS");
  console.log("═".repeat(70));

  // Overall accuracy
  const totalCorrect = results.filter((r) => r.correct).length;
  const totalQuestions = results.length;
  const overallAccuracy = (totalCorrect / totalQuestions) * 100;

  console.log(
    `\n  Overall Accuracy: ${totalCorrect}/${totalQuestions} (${overallAccuracy.toFixed(1)}%)\n`
  );

  // Per-type breakdown
  const typeGroups = new Map<string, EvalResult[]>();
  for (const r of results) {
    if (!typeGroups.has(r.questionType)) {
      typeGroups.set(r.questionType, []);
    }
    typeGroups.get(r.questionType)!.push(r);
  }

  console.log("  By question type:");
  console.log("  " + "─".repeat(66));

  for (const [typeName, typeResults] of typeGroups) {
    const correct = typeResults.filter((r) => r.correct).length;
    const total = typeResults.length;
    const acc = (correct / total) * 100;
    const bar = "█".repeat(Math.round(acc / 5)) + "░".repeat(20 - Math.round(acc / 5));
    console.log(
      `  ${typeName.padEnd(30)} ${correct}/${total}  ${bar} ${acc.toFixed(0)}%`
    );
  }

  // Timing stats
  const avgIndexTime =
    results.reduce((s, r) => s + r.indexingTimeMs, 0) / results.length;
  const avgRetrievalTime =
    results.reduce((s, r) => s + r.retrievalTimeMs, 0) / results.length;
  const avgEntities =
    results.reduce((s, r) => s + r.stats.entities, 0) / results.length;
  const avgFacts =
    results.reduce((s, r) => s + r.stats.facts, 0) / results.length;

  console.log("\n  Performance:");
  console.log("  " + "─".repeat(66));
  console.log(`  Avg indexing time:     ${(avgIndexTime / 1000).toFixed(1)}s per question`);
  console.log(`  Avg retrieval time:    ${(avgRetrievalTime / 1000).toFixed(1)}s per question`);
  console.log(`  Avg entities per KG:   ${avgEntities.toFixed(0)}`);
  console.log(`  Avg facts per KG:      ${avgFacts.toFixed(0)}`);

  console.log("\n" + "═".repeat(70) + "\n");

  // Write detailed results to file
  const outputPath = `data/eval_results_${dataset}_${new Date().toISOString().slice(0, 10)}.json`;
  await Bun.write(outputPath, JSON.stringify(results, null, 2));
  console.log(`Detailed results saved to ${outputPath}`);
}

main().catch((err) => {
  console.error("Evaluation failed:", err);
  process.exit(1);
});
