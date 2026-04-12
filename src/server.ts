/**
 * server.ts — API server for the eval visualization frontend.
 *
 * Provides endpoints to:
 *   - Browse the LongMemEval dataset
 *   - Trigger eval runs (full sweep or single question)
 *   - Stream results via Server-Sent Events (SSE)
 *
 * SSE streaming means the frontend gets results in real-time as each
 * question completes, rather than waiting for the entire eval to finish.
 *
 * Usage:
 *   bun run src/server.ts
 *
 * Then open http://localhost:3001 (or use the Vite dev server with proxy).
 */

import {
  loadDataset,
  selectItems,
  evaluateQuestion,
  type EvalItem,
  type EvalResult,
} from "./eval.ts";

const PORT = 3001;

/** Currently loaded dataset (cached after first load) */
let cachedDataset: EvalItem[] | null = null;

async function getDataset(): Promise<EvalItem[]> {
  if (!cachedDataset) {
    cachedDataset = await loadDataset("oracle");
  }
  return cachedDataset;
}

/**
 * Create an SSE response that streams eval results as they complete.
 *
 * SSE (Server-Sent Events) is a simple protocol where the server sends
 * `data: ...\n\n` lines to the client over a long-lived HTTP connection.
 * The browser's EventSource API handles reconnection automatically.
 *
 * We send three event types:
 *   - "progress": { completed, total, result } — after each question
 *   - "done": { results } — when all questions are finished
 *   - "error": { message } — if something goes wrong
 */
function createSSEStream(
  items: EvalItem[]
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();

  return new ReadableStream({
    async start(controller) {
      const results: EvalResult[] = [];

      // Send initial event with total count
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({ type: "start", total: items.length })}\n\n`
        )
      );

      for (let i = 0; i < items.length; i++) {
        try {
          const result = await evaluateQuestion(items[i]!);
          results.push(result);

          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: "progress",
                completed: i + 1,
                total: items.length,
                result,
              })}\n\n`
            )
          );
        } catch (error) {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                type: "error",
                message: `Failed on question ${items[i]?.question_id}: ${error}`,
                completed: i + 1,
                total: items.length,
              })}\n\n`
            )
          );
        }
      }

      // Send completion event
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({ type: "done", results })}\n\n`
        )
      );

      controller.close();
    },
  });
}

/** CORS headers for dev (Vite runs on a different port) */
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    // Handle CORS preflight
    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    // ── GET /api/dataset — list all questions in the dataset ──
    if (url.pathname === "/api/dataset") {
      try {
        const dataset = await getDataset();

        // Return lightweight question list (no haystack sessions)
        const questions = dataset.map((item) => ({
          questionId: item.question_id,
          questionType: item.question_type,
          question: item.question,
          answer: item.answer,
          sessionCount: item.haystack_sessions.length,
        }));

        // Group by type for the frontend
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

    // ── GET /api/eval/run?count=2&type=temporal-reasoning — run eval sweep ──
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
          headers: {
            ...corsHeaders,
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      } catch (error) {
        return Response.json(
          { error: `Failed to start eval: ${error}` },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    // ── GET /api/eval/question?id=xxx — run a single question ──
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
          headers: {
            ...corsHeaders,
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      } catch (error) {
        return Response.json(
          { error: `Failed to evaluate question: ${error}` },
          { status: 500, headers: corsHeaders }
        );
      }
    }

    // Unknown route
    return Response.json(
      {
        error: "Not found",
        routes: [
          "GET /api/dataset",
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
console.log("  GET /api/dataset              — list all dataset questions");
console.log("  GET /api/eval/run?count=2     — run eval sweep (SSE)");
console.log("  GET /api/eval/question?id=... — run single question (SSE)");
