/**
 * server.ts — API server for the eval visualization frontend.
 *
 * Provides endpoints to:
 *   - Browse the LongMemEval dataset
 *   - Trigger eval runs (full sweep or single question)
 *   - Stream results via Server-Sent Events (SSE)
 *
 * SSE streams turn-level events so the frontend can show live progress
 * as each turn is processed, not just when entire questions complete.
 *
 * Usage:
 *   bun run src/server.ts
 */

import {
  loadDataset,
  selectItems,
  evaluateQuestion,
  countTurnPairs,
  type EvalItem,
  type EvalResult,
  type TurnSnapshot,
} from "./eval.ts";

const PORT = 3001;

let cachedDataset: EvalItem[] | null = null;

async function getDataset(): Promise<EvalItem[]> {
  if (!cachedDataset) {
    cachedDataset = await loadDataset("oracle");
  }
  return cachedDataset;
}

/**
 * Create an SSE stream that sends turn-level and question-level events.
 *
 * Event types:
 *   - "start": { totalQuestions, totalTurns }
 *   - "question_start": { questionIndex, questionId, questionType, totalTurns }
 *   - "turn": { questionIndex, snapshot, turnsDone, totalTurns }
 *   - "question_done": { questionIndex, result }
 *   - "error": { message, questionIndex }
 *   - "done": { results }
 */
function createSSEStream(items: EvalItem[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  // Pre-compute total turns across all questions
  const turnsPerQuestion = items.map(countTurnPairs);
  const totalTurnsAll = turnsPerQuestion.reduce((a, b) => a + b, 0);

  return new ReadableStream({
    async start(controller) {
      const send = (data: Record<string, unknown>) => {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
        );
      };

      send({
        type: "start",
        totalQuestions: items.length,
        totalTurns: totalTurnsAll,
      });

      const results: EvalResult[] = [];
      let globalTurnsDone = 0;

      for (let qi = 0; qi < items.length; qi++) {
        const item = items[qi]!;
        const questionTotalTurns = turnsPerQuestion[qi]!;

        send({
          type: "question_start",
          questionIndex: qi,
          questionId: item.question_id,
          questionType: item.question_type,
          question: item.question,
          totalTurns: questionTotalTurns,
        });

        try {
          const result = await evaluateQuestion(
            item,
            // onTurn callback — fires after each turn
            (snapshot: TurnSnapshot, totalTurns: number) => {
              globalTurnsDone++;
              send({
                type: "turn",
                questionIndex: qi,
                snapshot,
                turnsDone: snapshot.turnIndex + 1,
                totalTurns,
                globalTurnsDone,
                globalTotalTurns: totalTurnsAll,
              });
            }
          );

          results.push(result);

          send({
            type: "question_done",
            questionIndex: qi,
            completed: qi + 1,
            total: items.length,
            result,
          });
        } catch (error) {
          send({
            type: "error",
            message: `Failed on question ${item.question_id}: ${error}`,
            questionIndex: qi,
          });
        }
      }

      send({ type: "done", results });
      controller.close();
    },
  });
}

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const sseHeaders = {
  ...corsHeaders,
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  Connection: "keep-alive",
};

const server = Bun.serve({
  port: PORT,
  // Eval questions can take minutes — disable idle timeout for SSE streams
  idleTimeout: 0,

  async fetch(req) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    // ── GET /api/dataset ──
    if (url.pathname === "/api/dataset") {
      try {
        const dataset = await getDataset();

        const questions = dataset.map((item) => ({
          questionId: item.question_id,
          questionType: item.question_type,
          question: item.question,
          answer: item.answer,
          sessionCount: item.haystack_sessions.length,
        }));

        const types = [...new Set(dataset.map((i) => i.question_type))];

        return Response.json(
          { questions, types, total: dataset.length },
          { headers: corsHeaders }
        );
      } catch (error) {
        return Response.json(
          { error: `Failed to load dataset: ${error}` },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    // ── GET /api/dataset/question?id=xxx ──
    if (url.pathname === "/api/dataset/question") {
      try {
        const dataset = await getDataset();
        const questionId = url.searchParams.get("id");

        if (!questionId) {
          return Response.json(
            { error: "Missing 'id' parameter" },
            { status: 400, headers: corsHeaders }
          );
        }

        const item = dataset.find((i) => i.question_id === questionId);
        if (!item) {
          return Response.json(
            { error: `Question '${questionId}' not found` },
            { status: 404, headers: corsHeaders }
          );
        }

        const sessions = item.haystack_sessions.map((session, idx) => {
          const turns = Object.values(session) as {
            role: string;
            content: string;
          }[];
          return {
            sessionIndex: idx,
            turns: turns.map((t) => ({ role: t.role, content: t.content })),
          };
        });

        return Response.json(
          {
            questionId: item.question_id,
            questionType: item.question_type,
            question: item.question,
            answer: item.answer,
            questionDate: item.question_date,
            haystackDates: item.haystack_dates,
            answerSessionIds: item.answer_session_ids,
            sessions,
          },
          { headers: corsHeaders }
        );
      } catch (error) {
        return Response.json(
          { error: `Failed to load question: ${error}` },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    // ── GET /api/eval/run?count=2&type=... ──
    if (url.pathname === "/api/eval/run") {
      try {
        const dataset = await getDataset();

        const countParam = url.searchParams.get("count") ?? "2";
        const count =
          countParam === "all" ? ("all" as const) : parseInt(countParam, 10);
        const type = url.searchParams.get("type") ?? undefined;

        const items = selectItems(dataset, count, type);

        if (items.length === 0) {
          return Response.json(
            { error: "No questions match the criteria" },
            { status: 400, headers: corsHeaders }
          );
        }

        return new Response(createSSEStream(items), {
          headers: sseHeaders,
        });
      } catch (error) {
        return Response.json(
          { error: `Failed to start eval: ${error}` },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    // ── GET /api/eval/question?id=xxx ──
    if (url.pathname === "/api/eval/question") {
      try {
        const dataset = await getDataset();
        const questionId = url.searchParams.get("id");

        if (!questionId) {
          return Response.json(
            { error: "Missing 'id' parameter" },
            { status: 400, headers: corsHeaders }
          );
        }

        const item = dataset.find((i) => i.question_id === questionId);
        if (!item) {
          return Response.json(
            { error: `Question '${questionId}' not found` },
            { status: 404, headers: corsHeaders }
          );
        }

        return new Response(createSSEStream([item]), {
          headers: sseHeaders,
        });
      } catch (error) {
        return Response.json(
          { error: `Failed to evaluate question: ${error}` },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    return Response.json(
      {
        error: "Not found",
        routes: [
          "GET /api/dataset",
          "GET /api/dataset/question?id=...",
          "GET /api/eval/run?count=2&type=...",
          "GET /api/eval/question?id=...",
        ],
      },
      { status: 404, headers: corsHeaders }
    );
  },
});

console.log(`Eval API server running on http://localhost:${PORT}`);
console.log("Routes:");
console.log("  GET /api/dataset                  — list all dataset questions");
console.log("  GET /api/dataset/question?id=...  — full question with session turns");
console.log("  GET /api/eval/run?count=2         — run eval sweep (SSE)");
console.log("  GET /api/eval/question?id=...     — run single question (SSE)");
