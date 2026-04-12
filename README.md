# Brainy

A hippocampus-inspired dual-memory conversational agent built with LangGraph TypeScript and Bun. Implements a full [HippoRAG2](https://arxiv.org/abs/2502.14802) retrieval system with Personalized PageRank alongside a pressure-based conversation buffer, enabling long-form conversations with full-fidelity recent context and compressed long-term memory.

## Architecture

The agent uses a **pressure-based memory model**: real conversation turns accumulate in a buffer until they exceed a token threshold (~1024 tokens), at which point the buffer is summarized and indexed into HippoRAG2 for long-term retrieval. This means the LLM sees actual messages for recent turns and compressed summaries for older history.

```
User message
    │
    ▼
[retrieve]  ── query HippoRAG2 (PPR over knowledge graph) for long-term memories
    │
    ▼
[respond]   ── LLM call with: system prompt + conversation buffer + retrieved memories + current message
    │
    ▼
[memorize]  ── append exchange to buffer
    │           if buffer > 1024 tokens: summarize + index into HippoRAG2 + semantic forgetting
    ▼
   END
```

### 1. Conversation Buffer (Pressure-Based Memory)

Real conversation turns accumulate in a buffer, preserving full fidelity. When the buffer exceeds ~1024 tokens, it is compressed into a summary paragraph and indexed into HippoRAG2 for long-term retrieval. The summary replaces the buffer, and new turns accumulate on top of it until the next compression cycle. This mirrors how human short-term memory works — recent events are recalled in detail, older ones as gist.

### 2. HippoRAG2 Vector Memory

A full implementation of the [HippoRAG2 paper](https://arxiv.org/abs/2502.14802) (ICML 2025) as a standalone TypeScript module:

- **Knowledge Graph** with two node types (entity + passage) and three edge types (fact, passage, synonym)
- **OpenIE Triple Extraction** — LLM extracts (subject, predicate, object) triples from each conversation exchange
- **Synonym Detection** — KNN on entity embeddings creates edges between semantically similar entities (e.g., "ML" ↔ "machine learning")
- **Personalized PageRank** — power iteration over the composite graph for multi-hop retrieval
- **Recognition Memory** — LLM-based triple filtering before PPR (HippoRAG2's key improvement over v1)
- **Semantic Forgetting** — age-weighted pruning of low-salience passages when the store exceeds capacity
- **Dense Passage Retrieval (DPR)** — fallback when no triples match

#### How retrieval works

```
Query → embed → match triples (dense) → LLM filters triples (recognition memory)
  → extract seed entities → build PPR personalization vector
  → spread activation through knowledge graph → rank passage nodes → return top-K
```

The key insight: PPR finds passages connected to the query through **entity chains**, even if the passage text isn't directly similar. This enables multi-hop reasoning that plain vector search can't do.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | [Bun](https://bun.sh) |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraphjs/) (TypeScript) |
| LLM (responses + judging) | Claude Sonnet via `@langchain/anthropic` |
| LLM (recognition memory) | Claude Sonnet (quality-sensitive triple filtering) |
| LLM (extraction + summarisation) | Claude Haiku (OpenIE, compact memory) |
| Embeddings | [Voyage AI 3.5](https://docs.voyageai.com/) |
| Graph algorithm | Personalized PageRank (custom power iteration implementation) |

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

### Custom Multi-Turn Test

A synthetic 18-turn evaluation where the agent ingests 5 dense documents about a fictional robotics company (Aethon Robotics), handles 3 distractor turns, then answers 10 recall questions testing direct recall, multi-hop reasoning, entity linking, and detail recall.

```bash
bun run src/test.ts
```

**Result: 93% recall accuracy** across all 10 questions. Perfect on direct recall and multi-hop reasoning. Minor misses on exact numeric details.

| Test Type | Score |
|-----------|-------|
| Direct recall | 100% |
| Multi-hop | 100% |
| Entity linking | 90% |
| Detail recall | 83% |

### LongMemEval Benchmark (ICLR 2025)

Evaluated against [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — 500 questions across 5 long-term memory abilities with real multi-session chat histories.

```bash
# Download the dataset
mkdir -p data
wget -O data/longmemeval_oracle.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"

# Run eval (N per category, or --count all for full benchmark)
bun run src/eval.ts --count 5
```

#### Results (oracle dataset, 2 per category, 12 questions)

We ran the benchmark with three LLM configurations for the processing tasks (triple extraction, summarisation, recognition memory filtering):

| | Sonnet (all processing) | Haiku (all processing) | **Hybrid** (default) |
|---|---|---|---|
| **Recognition memory** | Sonnet | Haiku | Sonnet |
| **OpenIE + summarisation** | Sonnet | Haiku | Haiku |
| **Overall accuracy** | 83.3% (10/12) | 75.0% (9/12) | **83.3% (10/12)** |
| **Avg indexing time** | 61.3s/question | 28.0s/question | **26.3s/question** |
| **Avg retrieval time** | 3.0s/question | 2.8s/question | **1.9s/question** |
| **Avg KG size** | 52 entities, 51 facts | 53 entities, 50 facts | 52 entities, 49 facts |

The **hybrid config** (default) gives the best of both worlds: Sonnet-level accuracy at Haiku-level speed. Recognition memory is the quality-sensitive gate that decides which triples seed the PPR graph search, so Sonnet's stronger reasoning pays off there. OpenIE extraction and summarisation are more mechanical tasks where Haiku performs equally well.

**Breakdown by question type (hybrid config):**

| Category | Score | Description |
|----------|-------|-------------|
| Temporal reasoning | 2/2 (100%) | Understanding time-based relationships |
| Single-session (user) | 2/2 (100%) | Recalling user-stated facts |
| Single-session (assistant) | 2/2 (100%) | Recalling assistant-generated content |
| Single-session (preference) | 2/2 (100%) | Recalling user preferences |
| Knowledge update | 2/2 (100%) | Handling evolving/contradicting info |
| Multi-session reasoning | 0/2 (0%) | Synthesizing across sessions |

**Key findings:**
- **Hybrid config matches Sonnet accuracy while being 2.3x faster** — recognition memory is the only step where LLM quality matters for retrieval accuracy
- **Multi-session reasoning** (counting/aggregation across sessions) is the weakest category — all three configs fail on these. This is a top-K retrieval truncation issue: when the answer requires synthesizing information from 3+ separate passages, retrieving only top-3 may miss some
- **Single-session tasks** are near-perfect — the knowledge graph effectively indexes and retrieves facts from individual conversation sessions
- The knowledge graph grows to ~50 entities and ~50 facts per question on average, creating a rich retrieval structure even from relatively short conversations

## Project Structure

```
src/
  index.ts                     # Terminal chat loop (readline)
  graph.ts                     # LangGraph: START → retrieve → respond → memorize → END
  state.ts                     # State annotation (pressure-based buffer model)
  llm.ts                       # ChatAnthropic (Sonnet + Haiku) + Voyage 3.5
  utils.ts                     # Cosine similarity, hash IDs, normalization
  singletons.ts                # Shared HippoRAG + CompactMemory instances
  eval.ts                      # LongMemEval benchmark harness
  test.ts                      # Custom multi-turn evaluation
  nodes/
    retrieve.ts                # Query HippoRAG2 for relevant passages
    respond.ts                 # LLM response with memory-injected prompt
    memorize.ts                # Append to buffer + pressure-based summarize/index
  memory/
    compact-memory.ts          # Pressure-based buffer with token-threshold summarization
    types.ts                   # Memory context types
  hipporag/                    # Standalone HippoRAG2 module
    index.ts                   # Main class (index + retrieve + forget)
    knowledge-graph.ts         # Graph structure + Personalized PageRank
    embedding-store.ts         # In-memory embeddings + similarity search
    openie.ts                  # LLM-based triple extraction
    recognition-memory.ts      # LLM-based triple filtering
    types.ts                   # Triple, Passage, GraphNode, config types
```

## References

- [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG2)](https://arxiv.org/abs/2502.14802) — ICML 2025
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) — NeurIPS 2024
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) — ICLR 2025
