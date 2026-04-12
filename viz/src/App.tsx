import { useState, useEffect } from "react";
import type { EvalResult, DatasetQuestion } from "./types";
import { useEval } from "./useEval";
import Overview from "./Overview";
import QuestionDetail from "./QuestionDetail";
import EvalControls from "./EvalControls";
import DatasetBrowser from "./DatasetBrowser";

type View = "overview" | "dataset";

function App() {
  // Dataset state (loaded from API)
  const [datasetQuestions, setDatasetQuestions] = useState<DatasetQuestion[]>([]);
  const [datasetTypes, setDatasetTypes] = useState<string[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  // Results state (from eval runs or loaded file)
  const [results, setResults] = useState<EvalResult[]>([]);
  const [selected, setSelected] = useState<EvalResult | null>(null);
  const [view, setView] = useState<View>("overview");

  // Eval runner
  const eval_ = useEval();

  // Load dataset from API on mount
  useEffect(() => {
    fetch("/api/dataset")
      .then((r) => {
        if (!r.ok) throw new Error("API not available");
        return r.json();
      })
      .then((data) => {
        setDatasetQuestions(data.questions);
        setDatasetTypes(data.types);
      })
      .catch(() => {
        setDatasetError(
          "API server not running. Start it with: bun run src/server.ts"
        );
      });
  }, []);

  // Sync eval results into main results state
  useEffect(() => {
    if (eval_.results.length > 0) {
      setResults(eval_.results);
    }
  }, [eval_.results]);

  // Handle file upload (fallback when API not available)
  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        if (Array.isArray(data)) {
          setResults(data);
          setSelected(null);
        }
      } catch {
        alert("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  };

  // Detail view for a specific question
  if (selected) {
    return (
      <QuestionDetail
        result={selected}
        onBack={() => setSelected(null)}
      />
    );
  }

  const hasApi = datasetQuestions.length > 0;
  const hasResults = results.length > 0;

  return (
    <>
      <div className="header">
        <h1>Brainy Eval Dashboard</h1>
        <p>LongMemEval benchmark — evaluate and explore results</p>
      </div>

      {/* Tab navigation */}
      {hasApi && (
        <div className="tab-bar">
          <button
            className={`tab ${view === "overview" ? "active" : ""}`}
            onClick={() => setView("overview")}
          >
            Results {hasResults && `(${results.length})`}
          </button>
          <button
            className={`tab ${view === "dataset" ? "active" : ""}`}
            onClick={() => setView("dataset")}
          >
            Dataset ({datasetQuestions.length})
          </button>
        </div>
      )}

      {/* Eval controls (only if API is available) */}
      {hasApi && (
        <EvalControls
          types={datasetTypes}
          running={eval_.running}
          completed={eval_.completed}
          total={eval_.total}
          onRun={eval_.runSweep}
          onStop={eval_.stop}
        />
      )}

      {/* Error from eval */}
      {eval_.error && (
        <div className="error-banner">{eval_.error}</div>
      )}

      {/* Main content */}
      {view === "dataset" && hasApi ? (
        <DatasetBrowser
          questions={datasetQuestions}
          types={datasetTypes}
          running={eval_.running}
          onRunQuestion={(id) => {
            eval_.runQuestion(id);
            setView("overview");
          }}
        />
      ) : hasResults ? (
        <Overview results={results} onSelect={setSelected} />
      ) : (
        <div className="upload-zone">
          {datasetError && <p className="error-text">{datasetError}</p>}
          <p>
            {hasApi
              ? "Run an eval above, or upload previous results."
              : "Start the API server to run evals, or upload a results file."}
          </p>
          <label>
            Upload eval_results.json
            <input type="file" accept=".json" onChange={handleFile} />
          </label>
        </div>
      )}
    </>
  );
}

export default App;
