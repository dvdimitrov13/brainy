import { useState, useCallback, useRef } from "react";
import type { EvalResult, SSEEvent } from "./types";

export interface EvalState {
  /** Whether an eval is currently running */
  running: boolean;
  /** Progress: completed / total */
  completed: number;
  total: number;
  /** Results accumulated so far (live updates as questions complete) */
  results: EvalResult[];
  /** Error message if something went wrong */
  error: string | null;
}

/**
 * Hook for running evaluations via the SSE API.
 *
 * Returns the current eval state + functions to start/stop evals.
 * Results stream in real-time as each question completes.
 */
export function useEval() {
  const [state, setState] = useState<EvalState>({
    running: false,
    completed: 0,
    total: 0,
    results: [],
    error: null,
  });

  const abortRef = useRef<AbortController | null>(null);

  /**
   * Start an SSE eval stream from the given URL.
   * Used by both runSweep and runQuestion.
   */
  const startStream = useCallback(async (url: string) => {
    // Abort any existing run
    abortRef.current?.abort();
    const abort = new AbortController();
    abortRef.current = abort;

    setState({
      running: true,
      completed: 0,
      total: 0,
      results: [],
      error: null,
    });

    try {
      const response = await fetch(url, { signal: abort.signal });

      if (!response.ok) {
        const err = await response.json();
        setState((s) => ({
          ...s,
          running: false,
          error: err.error || "Request failed",
        }));
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        setState((s) => ({ ...s, running: false, error: "No response body" }));
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events (each ends with \n\n)
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? ""; // Keep incomplete part

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;

          try {
            const event: SSEEvent = JSON.parse(line.slice(6));

            if (event.type === "start") {
              setState((s) => ({ ...s, total: event.total }));
            } else if (event.type === "progress") {
              setState((s) => ({
                ...s,
                completed: event.completed,
                total: event.total,
                results: [...s.results, event.result],
              }));
            } else if (event.type === "error") {
              setState((s) => ({
                ...s,
                completed: event.completed,
                error: event.message,
              }));
            } else if (event.type === "done") {
              setState((s) => ({
                ...s,
                running: false,
                results: event.results,
              }));
            }
          } catch {
            // Skip malformed events
          }
        }
      }

      // Stream ended
      setState((s) => ({ ...s, running: false }));
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setState((s) => ({
        ...s,
        running: false,
        error: `Connection failed: ${err}`,
      }));
    }
  }, []);

  /** Run a full eval sweep with N questions per category */
  const runSweep = useCallback(
    (count: number | "all", type?: string) => {
      let url = `/api/eval/run?count=${count}`;
      if (type) url += `&type=${encodeURIComponent(type)}`;
      startStream(url);
    },
    [startStream]
  );

  /** Run a single question by ID */
  const runQuestion = useCallback(
    (questionId: string) => {
      startStream(`/api/eval/question?id=${encodeURIComponent(questionId)}`);
    },
    [startStream]
  );

  /** Stop the current eval run */
  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, running: false }));
  }, []);

  return { ...state, runSweep, runQuestion, stop };
}
