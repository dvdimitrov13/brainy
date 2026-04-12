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
        {/* Entity line */}
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
        {/* Fact line */}
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
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: 2,
              background: "var(--warn)",
              marginRight: 4,
              verticalAlign: "middle",
            }}
          />
          Facts
        </span>
        <span style={{ color: "var(--summarize)" }}>
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: 2,
              background: "var(--summarize)",
              marginRight: 4,
              verticalAlign: "middle",
            }}
          />
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

      {/* Two-phase retrieval pipeline visualization */}
      {result.retrieval ? (
        <div className="retrieval-pipeline">
          <h3>Retrieval Pipeline</h3>

          <div className="pipeline-phase">
            <div className="phase-header">
              <span className="phase-label">Phase 1: Triple Associations</span>
              <span className="phase-timing">
                {result.retrieval.tripleRetrievalMs}ms
              </span>
              <span className="phase-auto-badge">automatic</span>
            </div>
            {result.retrieval.triples.length > 0 ? (
              <div className="triple-list">
                {result.retrieval.triples.map((t, i) => (
                  <div key={i} className="triple-item">
                    <span className="triple-entity">{t.subject}</span>
                    <span className="triple-predicate">{t.predicate}</span>
                    <span className="triple-entity">{t.object}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="phase-empty">No relevant triples found</div>
            )}
          </div>

          <div className="pipeline-arrow">
            {result.retrieval.triples.length > 0
              ? "Agent calls recall_memory tool"
              : "Fallback to dense passage retrieval"}
          </div>

          <div className="pipeline-phase">
            <div className="phase-header">
              <span className="phase-label">Phase 2: Passage Recall (PPR)</span>
              <span className="phase-timing">
                {result.retrieval.passageRetrievalMs}ms
              </span>
              <span className="phase-tool-badge">tool call</span>
            </div>
            {result.retrieval.passages.length > 0 ? (
              <div className="passage-list">
                {result.retrieval.passages.map((p, i) => (
                  <div key={i} className="passage-item">
                    <span className="passage-idx">{i + 1}</span>
                    <span className="passage-text">
                      {p.text.slice(0, 200)}
                      {p.text.length > 200 ? "..." : ""}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="phase-empty">No passages retrieved</div>
            )}
          </div>

          <div className="pipeline-arrow">Chunk + rerank + pack within 1024 token budget</div>

          <div className="pipeline-phase">
            <div className="phase-header">
              <span className="phase-label">Reranked Context (what the LLM sees)</span>
              <span className="phase-timing">
                ~{Math.ceil((result.retrieval.rerankedContext?.length ?? 0) / 4)} tokens
              </span>
              <span className="phase-rerank-badge">reranked</span>
            </div>
            {result.retrieval.rerankedContext ? (
              <pre className="reranked-content">
                {result.retrieval.rerankedContext}
              </pre>
            ) : (
              <div className="phase-empty">No content after reranking</div>
            )}
          </div>
        </div>
      ) : (
        <div className="detail-cards">
          <div className="detail-card full-width">
            <h3>Retrieved Context (from HippoRAG)</h3>
            <pre>{result.retrievedContext || "(no passages retrieved)"}</pre>
          </div>
        </div>
      )}

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
