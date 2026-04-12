/** Snapshot of a single indexing turn */
export interface TurnSnapshot {
  turnIndex: number;
  sessionIndex: number;
  exchangeText: string;
  bufferTokensBefore: number;
  bufferTokensAfter: number;
  summarized: boolean;
  summaryText?: string;
  kgStats: { passages: number; entities: number; facts: number };
}

/** Result of evaluating a single question */
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

/** Lightweight question entry from the dataset API */
export interface DatasetQuestion {
  questionId: string;
  questionType: string;
  question: string;
  answer: string;
  sessionCount: number;
}

/** Dataset API response */
export interface DatasetResponse {
  questions: DatasetQuestion[];
  types: string[];
  total: number;
}

/** SSE event from the eval API */
export type SSEEvent =
  | { type: "start"; total: number }
  | { type: "progress"; completed: number; total: number; result: EvalResult }
  | { type: "error"; message: string; completed: number; total: number }
  | { type: "done"; results: EvalResult[] };
