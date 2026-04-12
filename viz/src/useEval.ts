import { useState, useCallback, useRef } from "react";
import type { EvalResult, TurnSnapshot, SSEEvent } from "./types";

/** Info about the currently executing question */
export interface ActiveQuestion {
  questionIndex: number;
  questionId: string;
  questionType: string;
  question: string;
  totalTurns: number;
  turnsDone: number;
  /** Turn snapshots accumulated so far for this question */
  liveSnapshots: TurnSnapshot[];
}

export interface EvalState {
  running: boolean;
  /** Questions completed / total */
  questionsCompleted: number;
  totalQuestions: number;
  /** Global turn progress across all questions */
  globalTurnsDone: number;
  globalTotalTurns: number;
  /** Currently executing question (null if between questions or idle) */
  activeQuestion: ActiveQuestion | null;
  /** Completed results */
  results: EvalResult[];
  error: string | null;
}

export function useEval() {
  const [state, setState] = useState<EvalState>({
    running: false,
    questionsCompleted: 0,
    totalQuestions: 0,
    globalTurnsDone: 0,
    globalTotalTurns: 0,
    activeQuestion: null,
    results: [],
    error: null,
  });

  const abortRef = useRef<AbortController | null>(null);

  const startStream = useCallback(async (url: string) => {
    abortRef.current?.abort();
    const abort = new AbortController();
    abortRef.current = abort;

    setState({
      running: true,
      questionsCompleted: 0,
      totalQuestions: 0,
      globalTurnsDone: 0,
      globalTotalTurns: 0,
      activeQuestion: null,
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

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;

          try {
            const event: SSEEvent = JSON.parse(line.slice(6));

            switch (event.type) {
              case "start":
                setState((s) => ({
                  ...s,
                  totalQuestions: event.totalQuestions,
                  globalTotalTurns: event.totalTurns,
                }));
                break;

              case "question_start":
                setState((s) => ({
                  ...s,
                  activeQuestion: {
                    questionIndex: event.questionIndex,
                    questionId: event.questionId,
                    questionType: event.questionType,
                    question: event.question,
                    totalTurns: event.totalTurns,
                    turnsDone: 0,
                    liveSnapshots: [],
                  },
                }));
                break;

              case "turn":
                setState((s) => ({
                  ...s,
                  globalTurnsDone: event.globalTurnsDone,
                  globalTotalTurns: event.globalTotalTurns,
                  activeQuestion: s.activeQuestion
                    ? {
                        ...s.activeQuestion,
                        turnsDone: event.turnsDone,
                        liveSnapshots: [
                          ...s.activeQuestion.liveSnapshots,
                          event.snapshot,
                        ],
                      }
                    : null,
                }));
                break;

              case "question_done":
                setState((s) => ({
                  ...s,
                  questionsCompleted: event.completed,
                  results: [...s.results, event.result],
                  activeQuestion: null,
                }));
                break;

              case "error":
                setState((s) => ({
                  ...s,
                  error: event.message,
                }));
                break;

              case "done":
                setState((s) => ({
                  ...s,
                  running: false,
                  results: event.results,
                  activeQuestion: null,
                }));
                break;
            }
          } catch {
            // Skip malformed events
          }
        }
      }

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

  const runSweep = useCallback(
    (count: number | "all", type?: string) => {
      let url = `/api/eval/run?count=${count}`;
      if (type) url += `&type=${encodeURIComponent(type)}`;
      startStream(url);
    },
    [startStream]
  );

  const runQuestion = useCallback(
    (questionId: string) => {
      startStream(`/api/eval/question?id=${encodeURIComponent(questionId)}`);
    },
    [startStream]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, running: false, activeQuestion: null }));
  }, []);

  return { ...state, runSweep, runQuestion, stop };
}
