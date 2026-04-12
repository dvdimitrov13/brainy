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
