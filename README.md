# Brainy

A hippocampus-inspired conversational agent with long-term memory, built with LangGraph TypeScript and Bun. Implements a full [HippoRAG2](https://arxiv.org/abs/2502.14802) retrieval system with Personalized PageRank and a two-phase retrieval model where memory associations surface automatically but deep recall is agent-initiated.

## Architecture

The agent has two memory systems mirroring human cognition:

- **Conversation buffer** (short-term) — real turns within the current session, summarized under memory pressure
- **HippoRAG2 knowledge graph** (long-term) — the only memory that persists across sessions

Retrieval is two-phase: lightweight triple associations surface automatically every turn, but the agent must actively decide to "recall" full passages via a tool call.

```
User message
    │
    ▼
[retrieve]  ── Phase 1: embed query → match triples → LLM filter (automatic)
    │
    ▼
[respond]   ── LLM sees: conversation buffer + filtered triples + current message
    │           If triples are relevant → calls recall_memory tool
    │           → Phase 2: PPR retrieval → chunk + rerank → pack within 1024 tokens
    │
    ▼
[memorize]  ── append exchange to buffer + track as pending
                if buffer > 1024 tokens: index each pending exchange separately
                into HippoRAG (parallel) + summarize buffer
                Session boundary: flush all pending into HippoRAG, reset buffer
    │
    ▼
   END
```

### Memory Model

**Within a session:**
- Real conversation turns accumulate in the buffer (full fidelity)
- When the buffer exceeds ~1024 tokens, memory pressure triggers:
  - Each pending exchange is indexed into HippoRAG as a separate passage (in parallel)
  - The buffer is summarized into a compact paragraph
  - New turns accumulate on top of the summary

**Between sessions:**
- All pending exchanges are flushed into HippoRAG
- The conversation buffer resets to empty
- HippoRAG is the only cross-session memory

### Two-Phase Retrieval

**Phase 1 (automatic, every turn):** Embed the query, find top-K triples by cosine similarity on fact embeddings, filter through recognition memory (LLM judge). Returns lightweight entity associations like `(alice, works_at, google)`. Fast and cheap.

**Phase 2 (agent-initiated, via tool call):** When the agent sees relevant triples and needs actual passage content, it calls the `recall_memory` tool. This triggers:
1. PPR over the knowledge graph using filtered triples as seeds
2. Chunking: split each retrieved exchange into ~256-token chunks (recursive character splitting, respecting user/assistant boundaries)
3. Reranking: Voyage `rerank-2` scores all chunks against the query
4. Packing: fill up to 1024 tokens with the highest-ranked chunks

This ensures the agent gets the most relevant snippets tightly packed, never blowing up the context window.

### HippoRAG2 Knowledge Graph

A full implementation of the [HippoRAG2 paper](https://arxiv.org/abs/2502.14802) (ICML 2025) as a standalone TypeScript module:

- **Knowledge Graph** with two node types (entity + passage) and three edge types (fact, passage, synonym)
- **OpenIE Triple Extraction** — LLM extracts (subject, predicate, object) triples from each exchange
- **Synonym Detection** — KNN on entity embeddings creates edges between semantically similar entities
- **Personalized PageRank** — power iteration over the composite graph for multi-hop retrieval
- **Recognition Memory** — LLM-based triple filtering before PPR (HippoRAG2's key v2 improvement)
- **Semantic Forgetting** — age-weighted pruning of low-salience passages when the store exceeds capacity
- **Dense Passage Retrieval (DPR)** — fallback when no triples match

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | [Bun](https://bun.sh) |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraphjs/) (TypeScript) |
| LLM (responses + judging) | Claude Sonnet via `@langchain/anthropic` |
| LLM (recognition memory) | Claude Sonnet (quality-sensitive triple filtering) |
| LLM (extraction + summarisation) | Claude Haiku (OpenIE, compact memory) |
| Embeddings | [Voyage AI 3.5](https://docs.voyageai.com/) |
| Reranking | [Voyage rerank-2](https://docs.voyageai.com/) |
| Graph algorithm | Personalized PageRank (custom power iteration) |

## Setup

```bash
# Install dependencies
bun install

# Add your API keys to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "VOYAGE_API_KEY=pa-..." >> .env

# Start chatting
bun run start
```

Type `stats` during conversation to see memory stats, `quit` to exit.

## Evaluation

### Eval Dashboard

An interactive React frontend for running and exploring evaluations.

```bash
# Terminal 1: API server (handles eval execution)
bun run server

# Terminal 2: Frontend with HMR
bun run viz
```

Features:
- **Run evals** from the UI — select count per category, optional type filter
- **Run individual questions** — browse the dataset, pick any question, run it in isolation
- **Live progress** — turn-by-turn streaming via SSE as each question executes
- **Drill into results** — per-question detail view with:
  - Two-phase retrieval pipeline visualization (triples → raw passages → reranked chunks)
  - Buffer token chart with peak + compression markers
  - KG growth chart (entities + facts over time)
  - Expandable turn-by-turn indexing history
- **Dataset browser** — search/filter all 500 LongMemEval questions, preview session turns

### LongMemEval Benchmark (ICLR 2025)

Evaluated against [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — 500 questions across 6 long-term memory abilities with real multi-session chat histories.

```bash
# Download the dataset
mkdir -p data
wget -O data/longmemeval_oracle.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"

# Run eval via CLI
bun run eval -- --count 2

# Or use the dashboard (bun run server + bun run viz)
```

### Custom Multi-Turn Test

A synthetic 18-turn evaluation with 5 dense documents, 3 distractors, and 10 recall questions.

```bash
bun run src/test.ts
```

## Project Structure

```
src/
  index.ts                     # Terminal chat loop (readline)
  graph.ts                     # LangGraph: START → retrieve → respond → memorize → END
  state.ts                     # State annotation (pressure-based buffer + pending exchanges)
  llm.ts                       # ChatAnthropic (Sonnet + Haiku + mid-tier) + Voyage 3.5
  chunking.ts                  # Chunk, rerank (Voyage rerank-2), and pack within token budget
  utils.ts                     # Cosine similarity, hash IDs, normalization
  singletons.ts                # Shared HippoRAG + CompactMemory instances
  server.ts                    # API server (Bun.serve) with SSE for eval streaming
  eval.ts                      # LongMemEval benchmark harness (CLI + importable)
  test.ts                      # Custom multi-turn evaluation
  nodes/
    retrieve.ts                # Phase 1: triple retrieval + recognition memory filter
    respond.ts                 # LLM response with recall_memory tool (Phase 2)
    memorize.ts                # Append to buffer + pressure-based index (per-exchange)
  memory/
    compact-memory.ts          # Pressure-based buffer with token-threshold summarization
    types.ts                   # Memory context types
  hipporag/                    # Standalone HippoRAG2 module
    index.ts                   # Main class (index + retrieveTriples + retrievePassages + forget)
    knowledge-graph.ts         # Graph structure + Personalized PageRank
    embedding-store.ts         # In-memory embeddings + similarity search
    openie.ts                  # LLM-based triple extraction
    recognition-memory.ts      # LLM-based triple filtering
    types.ts                   # Triple, Passage, GraphNode, config types
viz/                           # Eval dashboard (Vite + React)
  src/
    App.tsx                    # Main app with tabs + eval controls
    Overview.tsx               # Results summary + category bars + question list
    QuestionDetail.tsx         # Deep dive: retrieval pipeline + charts + turn history
    DatasetBrowser.tsx         # Browse/search dataset + session preview
    LiveProgress.tsx           # Real-time turn-level progress during eval
    EvalControls.tsx           # Run/stop eval + count/type selectors
    useEval.ts                 # SSE hook for streaming eval results
```

## References

- [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG2)](https://arxiv.org/abs/2502.14802) — ICML 2025
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) — NeurIPS 2024
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) — ICLR 2025
