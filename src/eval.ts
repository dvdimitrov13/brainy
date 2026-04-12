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
import type { NoteOperation } from "./memory/notepad.ts";
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
  notepadStats: { notepadTokens: number; exchangeCount: number; sectionCount: number };
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
  stats: { notepadTokens: number; exchangeCount: number; sectionCount: number };
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
          "Add new notes to your notepad. Only note top-level facts (user decisions, purchases, goals, dates, preferences — NOT generic advice). Keep 1-3 lines each, cite [exchange:ID]. Check TOC first to avoid duplicates.",
        parameters: {
          type: "object" as const,
          properties: {
            notes: {
              type: "array" as const,
              items: {
                type: "object" as const,
                properties: {
                  sectionPath: { type: "string" as const, description: "Hierarchical path: Topic/Subtopic" },
                  content: { type: "string" as const, description: "Note content with citations [exchange:ID]" },
                  afterSection: { type: "string" as const, description: "Optional: place after this section" },
                },
                required: ["sectionPath", "content"],
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
        description: "Update an existing section. Must read it first with read_notes.",
        parameters: {
          type: "object" as const,
          properties: {
            section: { type: "string" as const },
            content: { type: "string" as const },
          },
          required: ["section", "content"],
        },
      },
    },
    {
      type: "function" as const,
      function: {
        name: "read_notes",
        description: "Read a section's content (not subsections). Omit section to get TOC.",
        parameters: {
          type: "object" as const,
          properties: {
            section: { type: "string" as const, description: "Section heading to read. Omit for TOC." },
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
// TOOL EXECUTION (shared between indexing + answering)
// ══════════════════════════════════════════════

function executeToolCall(
  notepad: NotepadMemory,
  toolName: string,
  args: Record<string, unknown>
): string {
  switch (toolName) {
    case "write_notes": {
      const notes = args.notes as { sectionPath: string; content: string; afterSection?: string }[];
      const ops: NoteOperation[] = notes.map((n) => ({
        sectionPath: n.sectionPath,
        content: n.content,
        afterSection: n.afterSection,
      }));
      notepad.writeNotes(ops);
      return `Added ${ops.length} note(s). TOC:\n${notepad.getTOC()}`;
    }
    case "edit_notes": {
      const result = notepad.editSection(args.section as string, args.content as string);
      return result.ok ? `Updated "${args.section}".` : result.error!;
    }
    case "read_notes": {
      if (!args.section) return `Table of Contents:\n${notepad.getTOC()}`;
      const content = notepad.readSection(args.section as string);
      return content !== null
        ? content || "(section exists but has no direct content)"
        : `Section "${args.section}" not found. TOC:\n${notepad.getTOC()}`;
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
        const toc = notepad.getTOC();
        const rollingSummary = notepad.getRollingSummary();

        const noteMessages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
          new SystemMessage(
            `You are a helpful assistant with a notepad for long-term memory.

Your notepad (table of contents):
${toc}

${rollingSummary ? `Conversation summary: ${rollingSummary}` : ""}

**Your memory buffer is full. You MUST call write_notes to process these pending exchanges into notes.**

Note-taking rules:
- Only note TOP-LEVEL FACTS: user decisions, purchases, goals, preferences, events, dates, numbers
- Do NOT note generic advice you gave — you can regenerate that anytime
- Keep each note to 1-3 lines — cite [exchange:ID] so you can recall details later
- Include dates/times from the session headers
- Check the TOC first — if a relevant section exists, use edit_notes to update it instead of creating duplicates
- Organize hierarchically: use consistent top-level categories with subtopics

Pending exchanges:
${pendingExchanges.map((e) => `[exchange:${e.id}]\n${e.text}`).join("\n\n")}

Process these into organized notes, then respond briefly.`
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
      const toc = notepad.getTOC();
      const rollingSummary = notepad.getRollingSummary();

      const noteMessages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
        new SystemMessage(
          `You are a helpful assistant with a notepad. Process these exchanges into notes.

Your notepad TOC:
${toc}

${rollingSummary ? `Summary: ${rollingSummary}` : ""}

Note-taking rules:
- Only note TOP-LEVEL FACTS: user decisions, purchases, goals, preferences, events, dates, numbers
- Do NOT note generic advice you gave — you can regenerate that anytime
- Keep each note to 1-3 lines — cite [exchange:ID] so you can recall details later
- Include dates/times from the session headers
- Check the TOC first — if a relevant section exists, use edit_notes to update it instead of creating duplicates
- Organize hierarchically: use consistent top-level categories with subtopics

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

  const toc = notepad.getTOC();
  const rollingSummary = notepad.getRollingSummary();

  const contextParts: string[] = [];
  if (rollingSummary) contextParts.push(`Conversation summary: ${rollingSummary}`);
  if (conversationBuffer) contextParts.push(`Recent conversation:\n${conversationBuffer}`);
  contextParts.push(`Your notepad (table of contents):\n${toc}`);

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(
      `You are a helpful, friendly assistant with a notepad for long-term memory.

You have four tools:
- **write_notes** — add new notes
- **edit_notes** — update an existing section (must read first)
- **read_notes** — read a section's content, or get the table of contents
- **recall_exchange** — fetch raw conversation from a citation [exchange:ID]

For recall questions, read relevant notepad sections. For counting/listing, read multiple sections thoroughly.

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
        `${mark} (${s.sectionCount}s/${s.notepadTokens}t, idx:${(result.indexingTimeMs / 1000).toFixed(0)}s, ret:${(result.retrievalTimeMs / 1000).toFixed(0)}s)`
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
