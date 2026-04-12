import { useState } from "react";
import type { ActiveQuestion } from "./useEval";

interface Props {
  activeQuestion: ActiveQuestion;
  questionsCompleted: number;
  totalQuestions: number;
  globalTurnsDone: number;
  globalTotalTurns: number;
}

export default function LiveProgress({
  activeQuestion,
  questionsCompleted,
  totalQuestions,
  globalTurnsDone,
  globalTotalTurns,
}: Props) {
  const [expanded, setExpanded] = useState(true);
  const [expandedTurn, setExpandedTurn] = useState<number | null>(null);

  const globalPct =
    globalTotalTurns > 0 ? (globalTurnsDone / globalTotalTurns) * 100 : 0;
  const questionPct =
    activeQuestion.totalTurns > 0
      ? (activeQuestion.turnsDone / activeQuestion.totalTurns) * 100
      : 0;

  return (
    <div className="live-progress">
      {/* Global progress bar */}
      <div className="live-global">
        <div className="live-global-info">
          <span>
            Question {questionsCompleted + 1} / {totalQuestions}
          </span>
          <span className="live-global-turns">
            {globalTurnsDone} / {globalTotalTurns} turns total
          </span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${globalPct}%` }} />
        </div>
      </div>

      {/* Current question progress */}
      <div
        className="live-question"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="live-question-header">
          <span className="live-question-type">
            {activeQuestion.questionType}
          </span>
          <span className="live-question-text">
            {activeQuestion.question}
          </span>
          <span className="live-question-turns">
            Turn {activeQuestion.turnsDone} / {activeQuestion.totalTurns}
          </span>
          <span className="expand-icon">{expanded ? "−" : "+"}</span>
        </div>
        <div className="progress-bar thin">
          <div
            className="progress-fill question-fill"
            style={{ width: `${questionPct}%` }}
          />
        </div>
      </div>

      {/* Live turn snapshots */}
      {expanded && activeQuestion.liveSnapshots.length > 0 && (
        <div className="live-turns">
          {activeQuestion.liveSnapshots.map((snap) => {
            const isExpanded = expandedTurn === snap.turnIndex;
            return (
              <div
                key={snap.turnIndex}
                className={`turn-item ${snap.summarized ? "summarized" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  setExpandedTurn(isExpanded ? null : snap.turnIndex);
                }}
              >
                <div className="turn-item-header">
                  <span className="turn-num">
                    Turn {snap.turnIndex + 1}
                  </span>
                  <span className="session-badge">
                    Session {snap.sessionIndex + 1}
                  </span>
                  {snap.summarized && (
                    <span className="summarize-badge">SUMMARIZED</span>
                  )}
                  <span className="kg-info">
                    {snap.notepadStats.sectionCount}s / {snap.notepadStats.notepadTokens}t
                  </span>
                  <span className="token-info">
                    {snap.bufferTokensBefore} &rarr;{" "}
                    {snap.summarized ? (
                      <><span className="peak-tokens">{snap.bufferTokensPeak}</span> &rarr; {snap.bufferTokensAfter}</>
                    ) : (
                      snap.bufferTokensAfter
                    )}{" "}
                    tokens
                  </span>
                </div>

                {isExpanded && (
                  <div className="turn-expand">
                    {snap.exchangeText}
                    {snap.summarized && snap.summaryText && (
                      <>
                        <span className="summary-label">
                          Summary produced:
                        </span>
                        {snap.summaryText}
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
