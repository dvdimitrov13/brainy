import { useState, useEffect } from "react";
import type { EvalResult } from "./types";
import Overview from "./Overview";
import QuestionDetail from "./QuestionDetail";

function App() {
  const [results, setResults] = useState<EvalResult[] | null>(null);
  const [selected, setSelected] = useState<EvalResult | null>(null);

  // Try to load the default eval results from public/
  useEffect(() => {
    fetch("/eval_results.json")
      .then((r) => {
        if (r.ok) return r.json();
        return null;
      })
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) setResults(data);
      })
      .catch(() => {});
  }, []);

  // Handle file upload
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

  // No data — show upload prompt
  if (!results) {
    return (
      <div className="upload-zone">
        <h2>Brainy Eval Viewer</h2>
        <p>
          Load eval results to visualize. Run{" "}
          <code>bun run src/eval.ts</code> to generate results, or upload
          an existing JSON file.
        </p>
        <label>
          Upload eval_results.json
          <input type="file" accept=".json" onChange={handleFile} />
        </label>
      </div>
    );
  }

  // Detail view
  if (selected) {
    return (
      <QuestionDetail
        result={selected}
        onBack={() => setSelected(null)}
      />
    );
  }

  // Overview
  return <Overview results={results} onSelect={setSelected} />;
}

export default App;
