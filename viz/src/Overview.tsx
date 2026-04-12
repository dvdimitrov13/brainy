import type { EvalResult } from "./types";

interface Props {
  results: EvalResult[];
  onSelect: (r: EvalResult) => void;
}

export default function Overview({ results, onSelect }: Props) {
  const total = results.length;
  const correct = results.filter((r) => r.correct).length;
  const accuracy = total > 0 ? (correct / total) * 100 : 0;
  const avgIndex =
    results.reduce((s, r) => s + r.indexingTimeMs, 0) / total / 1000;
  const avgRetrieval =
    results.reduce((s, r) => s + r.retrievalTimeMs, 0) / total / 1000;
  const avgEntities =
    results.reduce((s, r) => s + r.stats.entities, 0) / total;
  const avgFacts = results.reduce((s, r) => s + r.stats.facts, 0) / total;

  // Group by type
  const byType = new Map<string, EvalResult[]>();
  for (const r of results) {
    if (!byType.has(r.questionType)) byType.set(r.questionType, []);
    byType.get(r.questionType)!.push(r);
  }

  return (
    <>
      <div className="header">
        <h1>Brainy Eval Dashboard</h1>
        <p>
          LongMemEval benchmark results — {total} questions, {correct} correct
        </p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">Accuracy</div>
          <div className={`value ${accuracy >= 75 ? "pass" : "fail"}`}>
            {accuracy.toFixed(1)}%
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Questions</div>
          <div className="value">
            {correct}/{total}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">Avg Index Time</div>
          <div className="value">{avgIndex.toFixed(1)}s</div>
        </div>
        <div className="stat-card">
          <div className="label">Avg Retrieval</div>
          <div className="value">{avgRetrieval.toFixed(1)}s</div>
        </div>
        <div className="stat-card">
          <div className="label">Avg Entities</div>
          <div className="value">{avgEntities.toFixed(0)}</div>
        </div>
        <div className="stat-card">
          <div className="label">Avg Facts</div>
          <div className="value">{avgFacts.toFixed(0)}</div>
        </div>
      </div>

      <div className="category-section">
        <h2>Accuracy by Category</h2>
        {[...byType.entries()].map(([type, items]) => {
          const c = items.filter((r) => r.correct).length;
          const pct = (c / items.length) * 100;
          const barClass =
            pct === 100 ? "perfect" : pct === 0 ? "zero" : "";
          return (
            <div className="category-bar" key={type}>
              <span className="name">{type}</span>
              <div className="bar-track">
                <div
                  className={`bar-fill ${barClass}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="score">
                {c}/{items.length} ({pct.toFixed(0)}%)
              </span>
            </div>
          );
        })}
      </div>

      <div className="questions-section">
        <h2>All Questions</h2>
        {results.map((r) => (
          <div
            className="question-row"
            key={r.questionId}
            onClick={() => onSelect(r)}
          >
            <span className={`badge ${r.correct ? "pass" : "fail"}`}>
              {r.correct ? "PASS" : "FAIL"}
            </span>
            <span className="q-text">{r.question}</span>
            <span className="q-type">{r.questionType}</span>
            <span className="q-timing">
              {(r.indexingTimeMs / 1000).toFixed(0)}s /{" "}
              {(r.retrievalTimeMs / 1000).toFixed(0)}s
            </span>
          </div>
        ))}
      </div>
    </>
  );
}
