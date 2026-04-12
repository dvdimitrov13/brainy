import { useState } from "react";
import type { EvalResult, TurnSnapshot } from "./types";

interface Props {
  result: EvalResult;
  onBack: () => void;
}

/** Bar chart showing buffer token count at each turn */
function BufferChart({ turns }: { turns: TurnSnapshot[] }) {
  if (turns.length === 0) return null;

  const maxTokens = Math.max(
    1024,
    ...turns.map((t) => Math.max(t.bufferTokensPeak, t.bufferTokensAfter))
  );
  const chartHeight = 120;
  const thresholdY = chartHeight - (1024 / maxTokens) * chartHeight;

  return (
    <div className="buffer-chart">
      <h4>Buffer Tokens per Turn</h4>
      <div className="chart-area" style={{ height: chartHeight }}>
        <div
          className="chart-threshold"
          style={{ top: thresholdY }}
        >
          <span className="chart-threshold-label">1024 threshold</span>
        </div>
        {turns.map((t) => {
          // For summarized turns, show the peak (before compression) as a ghost bar
          // and the after (compressed) as the solid bar
          const peakHeight = (t.bufferTokensPeak / maxTokens) * chartHeight;
          const afterHeight = (t.bufferTokensAfter / maxTokens) * chartHeight;
          return (
            <div
              key={t.turnIndex}
              className="chart-bar-wrapper"
              title={
                t.summarized
                  ? `Turn ${t.turnIndex + 1}: peaked at ${t.bufferTokensPeak} → compressed to ${t.bufferTokensAfter} tokens`
                  : `Turn ${t.turnIndex + 1}: ${t.bufferTokensAfter} tokens`
              }
            >
              {t.summarized && (
                <div
                  className="chart-bar peak"
                  style={{ height: Math.max(2, peakHeight) }}
                />
              )}
              <div
                className={`chart-bar ${t.summarized ? "summarized" : ""}`}
                style={{ height: Math.max(2, t.summarized ? afterHeight : peakHeight) }}
              />
            </div>
          );
        })}
      </div>
      <div className="chart-legend">
        <span className="legend-normal">Normal</span>
        <span className="legend-summarized">Compressed</span>
        <span className="legend-peak">Peak (before compression)</span>
      </div>
    </div>
  );
}

/** KG growth chart */
function KGChart({ turns }: { turns: TurnSnapshot[] }) {
  if (turns.length === 0) return null;

  const maxVal = Math.max(
    1,
    ...turns.map((t) => Math.max(t.kgStats.entities, t.kgStats.facts))
  );
  const chartHeight = 100;

  return (
    <div className="buffer-chart">
      <h4>Knowledge Graph Growth</h4>
      <svg
        width="100%"
        height={chartHeight + 20}
        viewBox={`0 0 ${turns.length * 20} ${chartHeight + 20}`}
        preserveAspectRatio="none"
      >
        {/* Entities line */}
        <polyline
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          points={turns
            .map(
              (t, i) =>
                `${i * 20 + 10},${chartHeight - (t.kgStats.entities / maxVal) * chartHeight}`
            )
            .join(" ")}
        />
        {/* Facts line */}
        <polyline
          fill="none"
          stroke="var(--warn)"
          strokeWidth="2"
          points={turns
            .map(
              (t, i) =>
                `${i * 20 + 10},${chartHeight - (t.kgStats.facts / maxVal) * chartHeight}`
            )
            .join(" ")}
        />
        {/* Summarization markers */}
        {turns
          .filter((t) => t.summarized)
          .map((t) => (
            <line
              key={t.turnIndex}
              x1={t.turnIndex * 20 + 10}
              x2={t.turnIndex * 20 + 10}
              y1={0}
              y2={chartHeight}
              stroke="var(--summarize)"
              strokeWidth="1"
              strokeDasharray="3,3"
              opacity={0.5}
            />
          ))}
      </svg>
      <div className="chart-legend">
        <span className="legend-normal">Entities</span>
        <span style={{ color: "var(--warn)" }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "var(--warn)", marginRight: 4, verticalAlign: "middle" }} />
          Facts
        </span>
        <span style={{ color: "var(--summarize)" }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "var(--summarize)", marginRight: 4, verticalAlign: "middle" }} />
          Summarization
        </span>
      </div>
    </div>
  );
}

export default function QuestionDetail({ result, onBack }: Props) {
  const [expandedTurn, setExpandedTurn] = useState<number | null>(null);

  return (
    <div className="detail-view">
      <button className="back-btn" onClick={onBack}>
        &larr; Back to overview
      </button>

      <div className="detail-header">
        <h2>
          <span className={`badge ${result.correct ? "pass" : "fail"}`}>
            {result.correct ? "PASS" : "FAIL"}
          </span>{" "}
          {result.question}
        </h2>
        <div className="detail-meta">
          <span>Type: {result.questionType}</span>
          <span>
            Index: {(result.indexingTimeMs / 1000).toFixed(1)}s
          </span>
          <span>
            Retrieval: {(result.retrievalTimeMs / 1000).toFixed(1)}s
          </span>
          <span>
            KG: {result.stats.entities}e / {result.stats.facts}f /{" "}
            {result.stats.passages}p
          </span>
          <span>Turns: {result.turns.length}</span>
        </div>
      </div>

      <div className="detail-cards">
        <div className="detail-card">
          <h3>Expected Answer</h3>
          <p>{result.expectedAnswer}</p>
        </div>
        <div className="detail-card">
          <h3>Generated Answer</h3>
          <p>{result.generatedAnswer}</p>
        </div>
      </div>

      {/* Agent tool call timeline */}
      {result.retrieval ? (
        <div className="retrieval-pipeline">
          <h3>
            Agent Memory Retrieval
            <span className="retrieval-total">
              {result.retrieval.totalRetrievalMs}ms total
              {result.retrieval.toolCalls.length === 0 && " — no tools called"}
            </span>
          </h3>

          {result.retrieval.toolCalls.length === 0 ? (
            <div className="phase-empty">
              Agent responded without consulting memory
            </div>
          ) : (
            <div className="tool-call-timeline">
              {result.retrieval.toolCalls.map((tc, i) => (
                <div key={i} className="tool-call-item">
                  <div className="phase-header">
                    <span className="phase-label">{tc.tool}()</span>
                    <span className="phase-desc">
                      query: "{tc.query.slice(0, 80)}
                      {tc.query.length > 80 ? "..." : ""}"
                    </span>
                    <span className="phase-timing">{tc.durationMs}ms</span>
                    <span className="phase-tool-badge">{tc.tool}</span>
                  </div>
                  <pre className="tool-call-result">{tc.result}</pre>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : null}

      {/* Retrieved context from tool calls */}
      <div className="detail-cards">
        <div className="detail-card full-width">
          <h3>Retrieved Context</h3>
          <pre className="notepad-content">
            {result.retrievedContext || "(no passages retrieved)"}
          </pre>
        </div>
      </div>

      {result.conversationBuffer && (
        <div className="detail-cards">
          <div className="detail-card full-width">
            <h3>Conversation Buffer (at question time)</h3>
            <pre>
              {result.conversationBuffer.slice(0, 1000)}
              {result.conversationBuffer.length > 1000 ? "\n..." : ""}
            </pre>
          </div>
        </div>
      )}

      {result.turns.length > 0 && (
        <div className="timeline-section">
          <h3>
            Turn-by-Turn Indexing ({result.turns.length} turns)
          </h3>

          <BufferChart turns={result.turns} />
          <KGChart turns={result.turns} />

          <div className="turn-list">
            {result.turns.map((turn) => {
              const isExpanded = expandedTurn === turn.turnIndex;
              return (
                <div
                  key={turn.turnIndex}
                  className={`turn-item ${turn.summarized ? "summarized" : ""}`}
                  onClick={() =>
                    setExpandedTurn(isExpanded ? null : turn.turnIndex)
                  }
                >
                  <div className="turn-item-header">
                    <span className="turn-num">
                      Turn {turn.turnIndex + 1}
                    </span>
                    <span className="session-badge">
                      Session {turn.sessionIndex + 1}
                    </span>
                    {turn.summarized && (
                      <span className="summarize-badge">
                        SUMMARIZED
                      </span>
                    )}
                    <span className="kg-info">
                      {turn.kgStats.entities}e / {turn.kgStats.facts}f
                    </span>
                    <span className="token-info">
                      {turn.bufferTokensBefore} &rarr;{" "}
                      {turn.summarized ? (
                        <><span className="peak-tokens">{turn.bufferTokensPeak}</span> &rarr; {turn.bufferTokensAfter}</>
                      ) : (
                        turn.bufferTokensAfter
                      )}{" "}
                      tokens
                    </span>
                  </div>

                  {isExpanded && (
                    <div className="turn-expand">
                      {turn.exchangeText}
                      {turn.summarized && turn.summaryText && (
                        <>
                          <span className="summary-label">
                            Summary produced:
                          </span>
                          {turn.summaryText}
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
