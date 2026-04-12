/** Snapshot of a single indexing turn */
export interface TurnSnapshot {
  turnIndex: number;
  sessionIndex: number;
  exchangeText: string;
  bufferTokensBefore: number;
  /** Token count after appending exchange but before summarization */
  bufferTokensPeak: number;
  bufferTokensAfter: number;
  summarized: boolean;
  summaryText?: string;
  kgStats: { passages: number; entities: number; facts: number };
}

/** A single tool call the agent made */
export interface ToolCallTrace {
  tool: "remember";
  query: string;
  result: string;
  durationMs: number;
}

/** Tracks the agent's retrieval decisions */
export interface RetrievalTrace {
  toolCalls: ToolCallTrace[];
  totalRetrievalMs: number;
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
  retrieval?: RetrievalTrace;
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

/** Full question detail from the dataset API (includes session turns) */
export interface QuestionDetail {
  questionId: string;
  questionType: string;
  question: string;
  answer: string;
  questionDate: string;
  haystackDates: string[];
  answerSessionIds: string[];
  sessions: {
    sessionIndex: number;
    turns: { role: string; content: string }[];
  }[];
}

/** SSE events from the eval API */
export type SSEEvent =
  | { type: "start"; totalQuestions: number; totalTurns: number }
  | {
      type: "question_start";
      questionIndex: number;
      questionId: string;
      questionType: string;
      question: string;
      totalTurns: number;
    }
  | {
      type: "turn";
      questionIndex: number;
      snapshot: TurnSnapshot;
      turnsDone: number;
      totalTurns: number;
      globalTurnsDone: number;
      globalTotalTurns: number;
    }
  | {
      type: "question_done";
      questionIndex: number;
      completed: number;
      total: number;
      result: EvalResult;
    }
  | { type: "error"; message: string; questionIndex: number }
  | { type: "done"; results: EvalResult[] };
