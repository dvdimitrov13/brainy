import { useState } from "react";
import type { DatasetQuestion, QuestionDetail } from "./types";

interface Props {
  questions: DatasetQuestion[];
  types: string[];
  running: boolean;
  onRunQuestion: (questionId: string) => void;
}

/** Shows the full session/turn breakdown for a question */
function SessionViewer({ detail }: { detail: QuestionDetail }) {
  const [expandedSession, setExpandedSession] = useState<number | null>(null);

  // Compute total turns and estimate buffer pressure points
  let buffer = "";
  const pressurePoints: { turnIndex: number; sessionIndex: number }[] = [];
  let globalTurn = 0;

  for (const session of detail.sessions) {
    for (let t = 0; t < session.turns.length; t += 2) {
      const user = session.turns[t];
      const assistant = session.turns[t + 1];
      let exchange = `User: ${user?.content ?? ""}`;
      if (assistant) exchange += `\nAssistant: ${assistant.content}`;

      buffer += (buffer ? "\n\n" : "") + exchange;
      const tokens = Math.ceil(buffer.length / 4);

      if (tokens > 1024) {
        pressurePoints.push({
          turnIndex: globalTurn,
          sessionIndex: session.sessionIndex,
        });
        buffer = "[Summary] compressed";
      }
      globalTurn++;
    }
  }

  const totalTurnPairs = globalTurn;

  return (
    <div className="session-viewer">
      <div className="session-summary">
        {detail.sessions.length} sessions, {totalTurnPairs} turn pairs,{" "}
        {pressurePoints.length} summarization points
      </div>

      {detail.sessions.map((session) => {
        const isExpanded = expandedSession === session.sessionIndex;
        const turnPairs: { user: string; assistant?: string }[] = [];
        for (let t = 0; t < session.turns.length; t += 2) {
          turnPairs.push({
            user: session.turns[t]?.content ?? "",
            assistant: session.turns[t + 1]?.content,
          });
        }

        return (
          <div key={session.sessionIndex} className="session-block">
            <div
              className="session-header"
              onClick={() =>
                setExpandedSession(isExpanded ? null : session.sessionIndex)
              }
            >
              <span className="session-label">
                Session {session.sessionIndex + 1}
              </span>
              <span className="session-turns">
                {turnPairs.length} turn pairs, {session.turns.length} messages
              </span>
              <span className="expand-icon">{isExpanded ? "−" : "+"}</span>
            </div>

            {isExpanded && (
              <div className="session-turns-list">
                {turnPairs.map((pair, i) => (
                  <div key={i} className="turn-pair">
                    <div className="turn-msg user-msg">
                      <span className="role-label">User</span>
                      <span className="msg-content">{pair.user}</span>
                    </div>
                    {pair.assistant && (
                      <div className="turn-msg assistant-msg">
                        <span className="role-label">Assistant</span>
                        <span className="msg-content">{pair.assistant}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
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
  const [questionDetail, setQuestionDetail] = useState<QuestionDetail | null>(
    null
  );
  const [loadingDetail, setLoadingDetail] = useState(false);

  const filtered = questions.filter((q) => {
    if (typeFilter && q.questionType !== typeFilter) return false;
    if (filter && !q.question.toLowerCase().includes(filter.toLowerCase()))
      return false;
    return true;
  });

  const handleExpand = async (questionId: string) => {
    if (expanded === questionId) {
      setExpanded(null);
      setQuestionDetail(null);
      return;
    }

    setExpanded(questionId);
    setQuestionDetail(null);
    setLoadingDetail(true);

    try {
      const res = await fetch(
        `/api/dataset/question?id=${encodeURIComponent(questionId)}`
      );
      if (res.ok) {
        const data = await res.json();
        setQuestionDetail(data);
      }
    } catch {
      // Silently fail — just won't show sessions
    } finally {
      setLoadingDetail(false);
    }
  };

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
              onClick={() => handleExpand(q.questionId)}
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

                {loadingDetail && (
                  <div className="detail-loading">Loading sessions...</div>
                )}

                {questionDetail &&
                  questionDetail.questionId === q.questionId && (
                    <SessionViewer detail={questionDetail} />
                  )}

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
