import { useState } from "react";

interface Props {
  types: string[];
  running: boolean;
  completed: number;
  total: number;
  onRun: (count: number | "all", type?: string) => void;
  onStop: () => void;
}

export default function EvalControls({
  types,
  running,
  completed,
  total,
  onRun,
  onStop,
}: Props) {
  const [count, setCount] = useState<string>("2");
  const [type, setType] = useState<string>("");

  const handleRun = () => {
    const c = count === "all" ? ("all" as const) : parseInt(count, 10);
    onRun(c, type || undefined);
  };

  const progress = total > 0 ? (completed / total) * 100 : 0;

  return (
    <div className="eval-controls">
      <div className="controls-row">
        <div className="control-group">
          <label>Per category</label>
          <select value={count} onChange={(e) => setCount(e.target.value)}>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="5">5</option>
            <option value="10">10</option>
            <option value="20">20</option>
            <option value="all">All</option>
          </select>
        </div>

        <div className="control-group">
          <label>Category</label>
          <select value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">All types</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        {running ? (
          <button className="stop-btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="run-btn" onClick={handleRun}>
            Run Eval
          </button>
        )}
      </div>

      {running && (
        <div className="progress-bar-container">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="progress-label">
            {completed} / {total}
          </span>
        </div>
      )}
    </div>
  );
}
