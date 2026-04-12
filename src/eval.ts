/**
 * eval.ts — LongMemEval benchmark evaluation logic.
 *
 * Exports core evaluation functions that can be used by both the CLI
 * and the API server. Uses the same notepad-based memory system and
 * tool-calling flow as the live agent.
 *
 * Usage (CLI):
 *   bun run src/eval.ts --count 2
 *   bun run src/eval.ts --count all
 *   bun run src/eval.ts --type multi-session --count 5
 */

import { NotepadMemory } from "./memory/notepad.ts";
import type { NoteEntry } from "./memory/notepad-ops.ts";
import { llm } from "./llm.ts";
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
  notepadStats: { notepadTokens: number; exchangeCount: number; noteCount: number };
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
  notepadContent: string;
  stats: { notepadTokens: number; exchangeCount: number; noteCount: number };
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
      `You are an evaluation judge. Given a question, an expected answer, and a generated answer, determine if the generated answer is correct.

The generated answer is CORRECT if:
- It contains the key information from the expected answer (semantic match, not exact string)
- It may contain additional details — that's fine
- It may use different wording — what matters is the factual content

The generated answer is INCORRECT if:
- It misses the key facts from the expected answer
- It contradicts the expected answer
- It says "I don't know" or similar when an answer exists

Respond with ONLY "CORRECT" or "INCORRECT", nothing else.`
    ),
    new HumanMessage(
      `Question: ${question}\n\nExpected answer: ${expectedAnswer}\n\nGenerated answer: ${generatedAnswer}\n\nVerdict:`
    ),
  ]);

  const verdict =
    typeof response.content === "string"
      ? response.content.trim().toUpperCase()
      : "";

  return verdict.includes("CORRECT") && !verdict.includes("INCORRECT");
}

// ══════════════════════════════════════════════
// TOOL DEFINITIONS (same as respond.ts)
// ══════════════════════════════════════════════

function getEvalTools() {
  return [
    {
      type: "function" as const,
      function: {
        name: "write_notes",
        description:
          "Add notes to your notepad. One note per fact, self-contained with date and citation. " +
          "Only note user facts — not generic advice. Keep 1-2 lines each.",
        parameters: {
          type: "object" as const,
          properties: {
            notes: {
              type: "array" as const,
              items: {
                type: "object" as const,
                properties: {
                  date: { type: "string" as const, description: "When this fact occurred" },
                  content: { type: "string" as const, description: "Self-contained factual note (1-2 lines)" },
                  citations: { type: "array" as const, items: { type: "string" as const }, description: "Source exchange IDs" },
                },
                required: ["date", "content", "citations"],
              },
            },
          },
          required: ["notes"],
        },
      },
    },
    {
      type: "function" as const,
      function: {
        name: "edit_notes",
        description: "Update an existing note by number. Must read it first with read_notes.",
        parameters: {
          type: "object" as const,
          properties: {
            note_number: { type: "number" as const, description: "Note number to edit" },
            content: { type: "string" as const, description: "Updated content" },
          },
          required: ["note_number", "content"],
        },
      },
    },
    {
      type: "function" as const,
      function: {
        name: "read_notes",
        description: "Read notes. Provide a query to search, or omit to read all notes.",
        parameters: {
          type: "object" as const,
          properties: {
            query: { type: "string" as const, description: "Keyword search. Omit to read all." },
          },
          required: [],
        },
      },
    },
    {
      type: "function" as const,
      function: {
        name: "recall_exchange",
        description: "Fetch raw conversation from a citation [exchange:ID].",
        parameters: {
          type: "object" as const,
          properties: {
            id: { type: "string" as const, description: 'Exchange ID, e.g., "sess0-turn3"' },
          },
          required: ["id"],
        },
      },
    },
  ];
}

// ══════════════════════════════════════════════
// TOOL EXECUTION
// ══════════════════════════════════════════════

function executeToolCall(
  notepad: NotepadMemory,
  toolName: string,
  args: Record<string, unknown>
): string {
  switch (toolName) {
    case "write_notes": {
      const notes = args.notes as { date: string; content: string; citations: string[] }[];
      const entries: NoteEntry[] = notes.map((n) => ({
        date: n.date,
        content: n.content,
        citations: n.citations,
      }));
      notepad.writeNotes(entries);
      return `Added ${entries.length} note(s). Notepad now has ${notepad.getStats().noteCount} notes.`;
    }
    case "edit_notes": {
      const result = notepad.editNote(args.note_number as number, args.content as string);
      return result.ok ? `Updated note #${args.note_number}.` : result.error!;
    }
    case "read_notes": {
      if (args.query) return notepad.search(args.query as string);
      return notepad.getAllNotes();
    }
    case "recall_exchange": {
      const exchange = notepad.getExchange(args.id as string);
      return exchange ?? `Exchange "${args.id}" not found.`;
    }
    default:
      return `Unknown tool: ${toolName}`;
  }
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
  const notepad = new NotepadMemory();
  let conversationBuffer = "";
  let pendingExchanges: { id: string; text: string }[] = [];
  const turns: TurnSnapshot[] = [];
  let globalTurnIndex = 0;

  const totalTurns = countTurnPairs(item);
  const indexStart = Date.now();
  const TOOLS = getEvalTools();

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

      const exchangeId = `sess${sessIdx}-turn${Math.floor(t / 2)}`;
      notepad.storeExchange(exchangeId, exchangeText);

      const bufferTokensBefore = estimateTokens(conversationBuffer);
      conversationBuffer = notepad.append(conversationBuffer, exchangeText);
      pendingExchanges.push({ id: exchangeId, text: exchangeText });

      const bufferTokensPeak = estimateTokens(conversationBuffer);

      let summarized = false;
      let summaryText: string | undefined;

      // Check memory pressure — force note-writing via LLM
      if (notepad.shouldSummarize(conversationBuffer) && pendingExchanges.length > 0) {
        // Build a note-writing prompt identical to the forced write_notes flow
        const toc = notepad.getIndex();
        const rollingSummary = notepad.getRollingSummary();

        const noteMessages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
          new SystemMessage(
            `You are a helpful assistant with a notepad for long-term memory.

Your notepad index:
${toc}

${rollingSummary ? `Conversation summary: ${rollingSummary}` : ""}

**Your memory buffer is full. You MUST call write_notes to process these pending exchanges into notes.**

Note-taking rules:
- Write one note per distinct fact — self-contained with enough context to be findable from any angle
- Include the date from the session header
- Only note facts about the user: decisions, purchases, goals, events, dates, numbers, preferences
- Do NOT note generic advice you gave — you can regenerate that anytime
- Keep each note to 1-2 lines — cite [exchange:ID] for detail recall later
- Check existing notes first (use read_notes) to avoid duplicates

Pending exchanges:
${pendingExchanges.map((e) => `[exchange:${e.id}]\n${e.text}`).join("\n\n")}

Process these into notes, then respond briefly.`
          ),
          new HumanMessage("Process the pending exchanges into notes."),
        ];

        // Let the LLM write notes (up to 3 tool calls)
        for (let i = 0; i < 3; i++) {
          const resp = await llm.invoke(noteMessages, { tools: TOOLS });
          const tc = resp.tool_calls;
          if (!tc || tc.length === 0) break;

          noteMessages.push(resp);
          for (const call of tc) {
            const result = executeToolCall(notepad, call.name, call.args as Record<string, unknown>);
            noteMessages.push(new ToolMessage({ tool_call_id: call.id ?? `idx_${i}`, content: result }));
          }
        }

        // Generate rolling summary and compress buffer
        await notepad.updateRollingSummary(conversationBuffer);
        summaryText = notepad.getRollingSummary();
        conversationBuffer = "[Summary]\n" + summaryText;
        pendingExchanges = [];
        notepad.resetReadTracking();
        summarized = true;
      }

      const bufferTokensAfter = estimateTokens(conversationBuffer);
      const notepadStats = notepad.getStats();

      const snapshot: TurnSnapshot = {
        turnIndex: globalTurnIndex++,
        sessionIndex: sessIdx,
        exchangeText,
        bufferTokensBefore,
        bufferTokensPeak,
        bufferTokensAfter,
        summarized,
        summaryText,
        notepadStats,
      };

      turns.push(snapshot);
      onTurn?.(snapshot, totalTurns);
    }

    // ── Session boundary: flush pending exchanges as notes ──
    if (pendingExchanges.length > 0) {
      const toc = notepad.getIndex();
      const rollingSummary = notepad.getRollingSummary();

      const noteMessages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
        new SystemMessage(
          `You are a helpful assistant with a notepad. Process these exchanges into notes.

Your notepad index:
${toc}

${rollingSummary ? `Summary: ${rollingSummary}` : ""}

Note-taking rules:
- Write one note per distinct fact — self-contained, findable from any angle
- Include the date from the session header
- Only note user facts: decisions, purchases, goals, events, dates, numbers, preferences
- Do NOT note generic advice — keep 1-2 lines each with [exchange:ID] citation
- Check existing notes (use read_notes) to avoid duplicates

Pending exchanges:
${pendingExchanges.map((e) => `[exchange:${e.id}]\n${e.text}`).join("\n\n")}

Call write_notes to save important information, then respond briefly.`
        ),
        new HumanMessage("Process these exchanges into notes."),
      ];

      for (let i = 0; i < 3; i++) {
        const resp = await llm.invoke(noteMessages, { tools: TOOLS });
        const tc = resp.tool_calls;
        if (!tc || tc.length === 0) break;
        noteMessages.push(resp);
        for (const call of tc) {
          const result = executeToolCall(notepad, call.name, call.args as Record<string, unknown>);
          noteMessages.push(new ToolMessage({ tool_call_id: call.id ?? `flush_${i}`, content: result }));
        }
      }

      pendingExchanges = [];
    }
    // Generate rolling summary before resetting buffer
    if (conversationBuffer) {
      await notepad.updateRollingSummary(conversationBuffer);
    }
    conversationBuffer = "";
    notepad.resetReadTracking();
    notepad.newSession();
  }

  const indexingTimeMs = Date.now() - indexStart;

  // ══════════════════════════════════════════════
  // ANSWER GENERATION (same tool-calling flow as live agent)
  // ══════════════════════════════════════════════

  const toc = notepad.getIndex();
  const rollingSummary = notepad.getRollingSummary();

  const contextParts: string[] = [];
  if (rollingSummary) contextParts.push(`Conversation summary: ${rollingSummary}`);
  if (conversationBuffer) contextParts.push(`Recent conversation:\n${conversationBuffer}`);
  contextParts.push(`Your notepad index:\n${toc}`);

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(
      `You are a helpful, friendly assistant with a notepad for long-term memory.

You have four tools:
- **write_notes** — add new notes (one per fact, with date and citation)
- **edit_notes** — update a note by number (must read first)
- **read_notes** — search by keyword, or read all notes
- **recall_exchange** — fetch raw conversation from a citation [exchange:ID]

For recall questions, use read_notes to search or read all. For counting/listing, read all notes.

Do NOT mention your notepad or tools. Just respond naturally.

--- Your Memory ---
${contextParts.join("\n\n")}
--- End Memory ---`
    ),
    new HumanMessage(item.question),
  ];

  const toolCallTraces: ToolCallTrace[] = [];
  const retrievalStart = Date.now();
  let generatedAnswer = "";

  for (let i = 0; i < 8; i++) {
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
      const result = executeToolCall(notepad, toolCall.name, toolCall.args as Record<string, unknown>);

      toolCallTraces.push({
        tool: toolCall.name,
        query: JSON.stringify(toolCall.args),
        result,
        durationMs: Date.now() - callStart,
      });

      messages.push(new ToolMessage({ tool_call_id: toolCall.id ?? `ans_${i}`, content: result }));
    }
  }

  const retrievalTimeMs = Date.now() - retrievalStart;

  if (!generatedAnswer) {
    generatedAnswer = "I'm having trouble recalling. Could you rephrase?";
  }

  const correct = await judgeAnswer(item.question, item.answer, generatedAnswer);
  const stats = notepad.getStats();

  return {
    questionId: item.question_id,
    questionType: item.question_type,
    question: item.question,
    expectedAnswer: item.answer,
    generatedAnswer,
    correct,
    retrievedContext: toolCallTraces.map((t) => t.result).join("\n\n"),
    conversationBuffer,
    notepadContent: notepad.getFullContent(),
    stats,
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
        `${mark} (${s.noteCount}s/${s.notepadTokens}t, idx:${(result.indexingTimeMs / 1000).toFixed(0)}s, ret:${(result.retrievalTimeMs / 1000).toFixed(0)}s)`
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
