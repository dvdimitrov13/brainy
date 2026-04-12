/**
 * eval.ts — LongMemEval benchmark evaluation logic.
 *
 * Uses HippoRAG with tag-based filtering. The agent decides when to
 * call remember(query, type?, topics?) via the same tool-calling loop
 * as the live agent.
 *
 * Usage (CLI):
 *   bun run src/eval.ts --count 2
 *   bun run src/eval.ts --count all
 *   bun run src/eval.ts --type multi-session --count 5
 */

import { HippoRAG } from "./hipporag/index.ts";
import { llm, llmFast } from "./llm.ts";
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";

// ══════════════════════════════════════════════
// TYPES
// ══════════════════════════════════════════════

interface Turn {
  role: "user" | "assistant";
  content: string;
  has_answer?: boolean;
}

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

export interface TurnSnapshot {
  turnIndex: number;
  sessionIndex: number;
  exchangeText: string;
  bufferTokensBefore: number;
  bufferTokensPeak: number;
  bufferTokensAfter: number;
  summarized: boolean;
  summaryText?: string;
  kgStats: { passages: number; entities: number; facts: number };
}

export interface ToolCallTrace {
  tool: string;
  query: string;
  result: string;
  durationMs: number;
}

export interface RetrievalTrace {
  toolCalls: ToolCallTrace[];
  totalRetrievalMs: number;
}

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
  retrieval: RetrievalTrace;
}

// ══════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function countTurnPairs(item: EvalItem): number {
  let count = 0;
  for (const session of item.haystack_sessions) {
    count += Math.ceil(Object.values(session).length / 2);
  }
  return count;
}

export type OnTurnCallback = (snapshot: TurnSnapshot, totalTurns: number) => void;

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
      `You are an evaluation judge. Determine if the generated answer is correct.

CORRECT if it contains the key information from the expected answer (semantic match).
INCORRECT if it misses key facts, contradicts, or says "I don't know".

Respond with ONLY "CORRECT" or "INCORRECT".`
    ),
    new HumanMessage(
      `Question: ${question}\n\nExpected: ${expectedAnswer}\n\nGenerated: ${generatedAnswer}\n\nVerdict:`
    ),
  ]);

  const verdict =
    typeof response.content === "string"
      ? response.content.trim().toUpperCase()
      : "";
  return verdict.includes("CORRECT") && !verdict.includes("INCORRECT");
}

// ══════════════════════════════════════════════
// ROLLING SUMMARY
// ══════════════════════════════════════════════

async function generateRollingSummary(
  prev: string,
  buffer: string
): Promise<string> {
  const response = await llmFast.invoke([
    {
      role: "system" as const,
      content: `Compress conversation context into a rolling summary (2-4 sentences).
Previous: ${prev || "(none)"}
Recent: ${buffer}
Respond with ONLY the summary.`,
    },
    { role: "user" as const, content: "Generate rolling summary." },
  ]);

  return typeof response.content === "string"
    ? response.content.trim()
    : (response.content as Array<{ type: string; text?: string }>)
        .filter((b) => b.type === "text")
        .map((b) => b.text ?? "")
        .join("")
        .trim();
}

// ══════════════════════════════════════════════
// TOOLS
// ══════════════════════════════════════════════

function getEvalTools() {
  return [
    {
      type: "function" as const,
      function: {
        name: "explore_topics",
        description:
          "Find which memory topics match your question. Returns relevant topic tags for filtering.",
        parameters: {
          type: "object" as const,
          properties: {
            request: { type: "string" as const, description: "What you're looking for" },
          },
          required: ["request"],
        },
      },
    },
    {
      type: "function" as const,
      function: {
        name: "remember",
        description:
          "Search long-term memory. Use type/topics filters for precise results. " +
          "For counting/listing, make multiple calls with different filters.",
        parameters: {
          type: "object" as const,
          properties: {
            query: { type: "string" as const, description: "Search query" },
            type: {
              type: "array" as const,
              items: { type: "string" as const },
              description: 'Filter: "event","decision","preference","fact","goal","plan"',
            },
            topics: {
              type: "array" as const,
              items: { type: "string" as const },
              description: "Filter: use exact topic names from the topic list",
            },
          },
          required: ["query"],
        },
      },
    },
  ];
}

// ══════════════════════════════════════════════
// DATASET HELPERS
// ══════════════════════════════════════════════

export async function loadDataset(dataset: string = "oracle"): Promise<EvalItem[]> {
  const dataPath = dataset === "s"
    ? "data/longmemeval_s_cleaned.json"
    : "data/longmemeval_oracle.json";
  return await Bun.file(dataPath).json();
}

export function selectItems(
  allItems: EvalItem[],
  count: number | "all",
  type?: string
): EvalItem[] {
  let items = type ? allItems.filter((i) => i.question_type === type) : allItems;

  if (count !== "all") {
    if (type) {
      items = items.slice(0, count);
    } else {
      const byType = new Map<string, EvalItem[]>();
      for (const item of items) {
        if (!byType.has(item.question_type)) byType.set(item.question_type, []);
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

// ══════════════════════════════════════════════
// CORE EVALUATION
// ══════════════════════════════════════════════

export async function evaluateQuestion(
  item: EvalItem,
  onTurn?: OnTurnCallback
): Promise<EvalResult> {
  const hipporag = new HippoRAG();
  let conversationBuffer = "";
  let pendingExchanges: string[] = [];
  let pressureCount = 0;
  const turns: TurnSnapshot[] = [];
  let globalTurnIndex = 0;

  const totalTurns = countTurnPairs(item);
  const LINT_EVERY = 3;
  const indexStart = Date.now();
  const TOKEN_THRESHOLD = 1024;

  for (let sessIdx = 0; sessIdx < item.haystack_sessions.length; sessIdx++) {
    const session = item.haystack_sessions[sessIdx]!;
    const sessionTurns = Object.values(session) as Turn[];
    const sessionDate = item.haystack_dates?.[sessIdx] ?? "";

    for (let t = 0; t < sessionTurns.length; t += 2) {
      const userTurn = sessionTurns[t];
      const assistantTurn = sessionTurns[t + 1];
      if (!userTurn) continue;

      let exchangeText = sessionDate ? `[Session: ${sessionDate}]\n` : "";
      exchangeText += `User: ${userTurn.content}`;
      if (assistantTurn) exchangeText += `\nAssistant: ${assistantTurn.content}`;

      const bufferTokensBefore = estimateTokens(conversationBuffer);

      conversationBuffer = conversationBuffer
        ? `${conversationBuffer}\n\n${exchangeText}`
        : exchangeText;
      pendingExchanges.push(exchangeText);

      const bufferTokensPeak = estimateTokens(conversationBuffer);

      let summarized = false;
      let summaryText: string | undefined;

      if (estimateTokens(conversationBuffer) > TOKEN_THRESHOLD && pendingExchanges.length > 0) {
        await Promise.all(pendingExchanges.map((ex) => hipporag.index(ex)));

        const prevSummary = conversationBuffer.startsWith("[Summary]")
          ? conversationBuffer.slice("[Summary]\n".length).split("\n\n")[0] ?? ""
          : "";
        const summary = await generateRollingSummary(prevSummary, conversationBuffer);
        summaryText = summary;
        conversationBuffer = `[Summary]\n${summary}`;
        pendingExchanges = [];
        summarized = true;
        hipporag.forget();

        // Lint topic tags every N pressure events
        pressureCount++;
        if (pressureCount % LINT_EVERY === 0) {
          await hipporag.lintTopics();
        }
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

    // Session boundary: flush + lint + reset
    if (pendingExchanges.length > 0) {
      await Promise.all(pendingExchanges.map((ex) => hipporag.index(ex)));
      pendingExchanges = [];
    }
    await hipporag.lintTopics();
    pressureCount = 0;
    if (conversationBuffer) {
      const prevSummary = conversationBuffer.startsWith("[Summary]")
        ? conversationBuffer.slice("[Summary]\n".length).split("\n\n")[0] ?? ""
        : "";
      await generateRollingSummary(prevSummary, conversationBuffer);
    }
    conversationBuffer = "";
    hipporag.forget();
  }

  const indexingTimeMs = Date.now() - indexStart;

  // ══════════════════════════════════════════════
  // ANSWER GENERATION (agent tool-calling loop)
  // ══════════════════════════════════════════════

  const TOOLS = getEvalTools();

  const stats = hipporag.getStats();
  const topics = hipporag.getTopics();
  const contextParts: string[] = [];
  if (conversationBuffer) contextParts.push(`Conversation:\n${conversationBuffer}`);
  if (stats.passages > 0) {
    contextParts.push(`Long-term memory: ${stats.passages} passages, ${stats.entities} entities`);
    if (topics.length > 0) {
      contextParts.push(`Available topics: ${topics.join(", ")}`);
    }
  }

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(
      `You are a helpful assistant with long-term memory.

You have two tools:
- **explore_topics(request)** — finds which memory topics match your question
- **remember(query, type?, topics?)** — searches past conversations with optional filters

Types: event, decision, preference, fact, goal, plan
Topics: use exact names from the available topics list.

For counting/listing: explore_topics first, then remember with each relevant topic.

Do NOT mention your tools. Respond naturally.

--- Memory ---
${contextParts.join("\n\n")}
--- End ---`
    ),
    new HumanMessage(item.question),
  ];

  const toolCallTraces: ToolCallTrace[] = [];
  const retrievalStart = Date.now();
  let generatedAnswer = "";

  for (let i = 0; i < 6; i++) {
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
      const callStart = Date.now();
      let result = "";

      if (toolCall.name === "explore_topics") {
        const args = toolCall.args as { request: string };
        const topicList = hipporag.getTopics();

        if (topicList.length === 0) {
          result = "No topics in memory yet.";
        } else {
          const resp = await llmFast.invoke([
            {
              role: "system" as const,
              content: `Given a request and topic list, return relevant topics as JSON: {"relevant_topics": ["t1","t2"]}. Be inclusive.`,
            },
            {
              role: "user" as const,
              content: `Request: ${args.request}\nTopics: ${topicList.join(", ")}`,
            },
          ]);

          const respText = typeof resp.content === "string" ? resp.content : "";
          try {
            const jsonStr = respText.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
            const parsed = JSON.parse(jsonStr) as { relevant_topics: string[] };
            result = parsed.relevant_topics.length > 0
              ? `Relevant topics: ${parsed.relevant_topics.join(", ")}`
              : "No matching topics found.";
          } catch {
            result = `Available topics: ${topicList.join(", ")}`;
          }
        }

        toolCallTraces.push({
          tool: "explore_topics",
          query: JSON.stringify(args),
          result,
          durationMs: Date.now() - callStart,
        });
      } else if (toolCall.name === "remember") {
        const args = toolCall.args as {
          query: string;
          type?: string[];
          topics?: string[];
        };

        const passages = await hipporag.retrieve(
          args.query,
          args.type,
          args.topics
        );

        result =
          passages.length > 0
            ? passages
                .map((p, idx) => {
                  const tagStr = `[${p.tags.type.join(",")}] [${p.tags.topics.join(",")}]`;
                  return `[Memory ${idx + 1}] ${tagStr}: ${p.text}`;
                })
                .join("\n\n")
            : "No relevant memories found.";

        toolCallTraces.push({
          tool: "remember",
          query: JSON.stringify(args),
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

  if (!generatedAnswer) {
    generatedAnswer = "I'm having trouble recalling. Could you rephrase?";
  }

  const correct = await judgeAnswer(item.question, item.answer, generatedAnswer);
  const finalStats = hipporag.getStats();

  return {
    questionId: item.question_id,
    questionType: item.question_type,
    question: item.question,
    expectedAnswer: item.answer,
    generatedAnswer,
    correct,
    retrievedContext: toolCallTraces.map((t) => t.result).join("\n\n"),
    conversationBuffer,
    stats: { passages: finalStats.passages, entities: finalStats.entities, facts: finalStats.facts },
    indexingTimeMs,
    retrievalTimeMs,
    turns,
    retrieval: { toolCalls: toolCallTraces, totalRetrievalMs: retrievalTimeMs },
  };
}

// ══════════════════════════════════════════════
// CLI
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

  for (let i = 0; i < items.length; i++) {
    const item = items[i]!;
    const prefix = `[${i + 1}/${items.length}]`;

    try {
      process.stdout.write(
        `${prefix} ${item.question_type.padEnd(28)} "${item.question.slice(0, 50)}..." `
      );

      const result = await evaluateQuestion(item);
      results.push(result);

      const mark = result.correct ? "PASS" : "FAIL";
      const s = result.stats;
      console.log(
        `${mark} (${s.entities}e/${s.facts}f/${s.passages}p, idx:${(result.indexingTimeMs / 1000).toFixed(0)}s, ret:${(result.retrievalTimeMs / 1000).toFixed(0)}s)`
      );

      if (!result.correct) {
        console.log(`       Expected: ${result.expectedAnswer}`);
        console.log(`       Got:      ${result.generatedAnswer.slice(0, 120)}...`);
      }
    } catch (error) {
      console.log(`ERROR: ${error}`);
    }
  }

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

const isMainModule = import.meta.path === Bun.main;
if (isMainModule) {
  main().catch((err) => {
    console.error("Evaluation failed:", err);
    process.exit(1);
  });
}
