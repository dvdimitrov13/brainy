import { useState } from "react";
import type { DatasetQuestion } from "./types";

interface Props {
  questions: DatasetQuestion[];
  types: string[];
  running: boolean;
  onRunQuestion: (questionId: string) => void;
}

export default function DatasetBrowser({
  questions,
  types,
  running,
  onRunQuestion,
}: Props) {
  const [filter, setFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = questions.filter((q) => {
    if (typeFilter && q.questionType !== typeFilter) return false;
    if (filter && !q.question.toLowerCase().includes(filter.toLowerCase()))
      return false;
    return true;
  });

  return (
    <div className="dataset-browser">
      <h2>Dataset Browser ({questions.length} questions)</h2>

      <div className="browser-filters">
        <input
          type="text"
          placeholder="Search questions..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="search-input"
        />
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="type-select"
        >
          <option value="">All types</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <span className="result-count">{filtered.length} shown</span>
      </div>

      <div className="browser-list">
        {filtered.slice(0, 50).map((q) => (
          <div
            key={q.questionId}
            className={`browser-item ${expanded === q.questionId ? "expanded" : ""}`}
          >
            <div
              className="browser-item-main"
              onClick={() =>
                setExpanded(expanded === q.questionId ? null : q.questionId)
              }
            >
              <span className="q-type-badge">{q.questionType}</span>
              <span className="q-text">{q.question}</span>
              <span className="q-sessions">
                {q.sessionCount} session{q.sessionCount !== 1 ? "s" : ""}
              </span>
            </div>

            {expanded === q.questionId && (
              <div className="browser-item-detail">
                <div className="detail-row">
                  <strong>Expected answer:</strong> {q.answer}
                </div>
                <div className="detail-row">
                  <strong>ID:</strong> {q.questionId}
                </div>
                <button
                  className="run-single-btn"
                  disabled={running}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRunQuestion(q.questionId);
                  }}
                >
                  {running ? "Running..." : "Run This Question"}
                </button>
              </div>
            )}
          </div>
        ))}
        {filtered.length > 50 && (
          <div className="browser-more">
            Showing 50 of {filtered.length} — narrow your search to see more
          </div>
        )}
      </div>
    </div>
  );
}
