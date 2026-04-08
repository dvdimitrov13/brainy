/**
 * index.ts — Main HippoRAG2 class that orchestrates all components.
 *
 * This is the top-level API for the HippoRAG2 module. It mirrors the
 * Python `HippoRAG.py` class and provides two main operations:
 *
 *   1. `index(text)` — Process a text passage and add it to the knowledge
 *      graph. Extracts triples, creates embeddings, builds graph edges.
 *
 *   2. `retrieve(query, topK)` — Given a query, find the most relevant
 *      stored passages using the full HippoRAG2 pipeline:
 *      dense retrieval → recognition memory → PPR graph search.
 *
 *   3. `forget()` — Semantic forgetting: prune old low-salience passages
 *      when the store exceeds capacity.
 *
 * Architecture mapping to the brain (from the HippoRAG paper):
 *   - Neocortex  → LLM (Claude Sonnet) — language understanding & extraction
 *   - Parahippocampal regions → Embedding model (Voyage 3.5) — pattern separation
 *   - Hippocampal index → Knowledge Graph + PPR — associative linking & retrieval
 *
 * TS note for Python devs:
 *   - `class HippoRAG { ... }` defines a class just like Python.
 *   - `private` fields are only accessible within the class (like _field in Python, but enforced).
 *   - `readonly` means the field can only be set in the constructor (like Python's @property with no setter).
 *   - `??` (nullish coalescing) is like `x if x is not None else default`.
 *   - `...spread` operator copies object/array elements (like {**dict} or [*list] in Python).
 */

import type { Triple, Passage, HippoRAGConfig } from "./types.ts";
import { DEFAULT_CONFIG } from "./types.ts";
import { KnowledgeGraph } from "./knowledge-graph.ts";
import { EmbeddingStore } from "./embedding-store.ts";
import { extractTriples, tripleToString } from "./openie.ts";
import { filterTriples } from "./recognition-memory.ts";
import { embedTexts, embedQuery } from "../llm.ts";
import {
  computeHashId,
  normalizeEntity,
  cosineSimilarity,
  minMaxNormalize,
} from "../utils.ts";

export class HippoRAG {
  // ──────────────────────────────────────────────
  // Configuration
  // ──────────────────────────────────────────────
  private readonly config: HippoRAGConfig;

  // ──────────────────────────────────────────────
  // Core components
  // ──────────────────────────────────────────────

  /** The knowledge graph (entity + passage nodes, weighted edges, PPR) */
  private graph: KnowledgeGraph = new KnowledgeGraph();

  /** Embedding store for passage (chunk) texts */
  private chunkStore: EmbeddingStore = new EmbeddingStore();

  /** Embedding store for entity (phrase) names */
  private entityStore: EmbeddingStore = new EmbeddingStore();

  /** Embedding store for fact (triple) strings */
  private factStore: EmbeddingStore = new EmbeddingStore();

  // ──────────────────────────────────────────────
  // Index mappings (mirrors HippoRAG Python)
  // ──────────────────────────────────────────────

  /** Map from entity node ID → set of passage IDs that mention this entity.
   *  Used to normalize phrase weights in PPR personalisation vector.
   *  Python equivalent: dict[str, set[str]] */
  private entityToChunkIds: Map<string, Set<string>> = new Map();

  /** Map from fact store ID → the actual Triple object.
   *  Used to look up triple content during retrieval. */
  private factIdToTriple: Map<string, Triple> = new Map();

  /** All indexed passages, keyed by passage ID */
  private passages: Map<string, Passage> = new Map();

  /** Auto-incrementing counter for passage IDs */
  private nextPassageIndex = 0;

  // ──────────────────────────────────────────────
  // Constructor
  // ──────────────────────────────────────────────

  /**
   * Create a new HippoRAG instance.
   *
   * @param config — optional partial config (unset fields use defaults)
   *
   * TS note: `Partial<T>` makes all fields of T optional.
   * Python equivalent: **kwargs with defaults.
   */
  constructor(config?: Partial<HippoRAGConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ══════════════════════════════════════════════
  // INDEXING PIPELINE
  // ══════════════════════════════════════════════

  /**
   * Index a new text passage into the HippoRAG knowledge graph.
   *
   * This is the full indexing pipeline from the HippoRAG2 paper:
   *   1. Extract triples via OpenIE (LLM)
   *   2. Embed the passage text
   *   3. Embed new entity names
   *   4. Embed fact (triple) strings
   *   5. Add passage node to graph
   *   6. Add fact edges (entity ↔ entity)
   *   7. Add passage edges (passage → entity)
   *   8. Add synonym edges (entity ↔ entity by embedding similarity)
   *
   * @param text — the conversation text to index
   * @returns the created Passage object
   */
  async index(text: string): Promise<Passage> {
    // ── Step 1: Extract triples via LLM ──
    const { triples, salience } = await extractTriples(text);

    // ── Step 2: Create passage ID ──
    const passageId = computeHashId(
      `passage-${this.nextPassageIndex++}-${text.slice(0, 50)}`,
      "passage-"
    );

    // ── Step 3: Process each triple — collect entities and facts ──
    // We collect everything BEFORE embedding so we can batch all texts
    // into a SINGLE Voyage API call (crucial for rate limit compliance).
    const newEntityNames: { id: string; name: string }[] = [];
    const newFactStrings: { id: string; text: string }[] = [];

    for (const triple of triples) {
      const subjectId = computeHashId(triple.subject, "entity-");
      const objectId = computeHashId(triple.object, "entity-");
      const factStr = tripleToString(triple);
      const factId = computeHashId(factStr, "fact-");

      // Track which entities are new (need embedding)
      // Deduplicate: only add if not already in our newEntityNames list
      if (
        !this.entityStore.has(subjectId) &&
        !newEntityNames.some((e) => e.id === subjectId)
      ) {
        newEntityNames.push({ id: subjectId, name: triple.subject });
      }
      if (
        !this.entityStore.has(objectId) &&
        !newEntityNames.some((e) => e.id === objectId)
      ) {
        newEntityNames.push({ id: objectId, name: triple.object });
      }

      // Track new facts (deduplicated)
      if (
        !this.factStore.has(factId) &&
        !newFactStrings.some((f) => f.id === factId)
      ) {
        newFactStrings.push({ id: factId, text: factStr });
        this.factIdToTriple.set(factId, triple);
      }
    }

    // ── Step 4: SINGLE batch embed call for passage + entities + facts ──
    // Combine all texts into one API call to minimize rate limit usage.
    // We'll split the results back out by index position.
    const allTextsToEmbed: string[] = [
      text, // index 0 = passage
      ...newEntityNames.map((e) => e.name), // indices 1..N = entities
      ...newFactStrings.map((f) => f.text), // indices N+1..M = facts
    ];

    const allEmbeddings = await embedTexts(allTextsToEmbed);

    // Split embeddings back out
    const passageEmbedding = allEmbeddings[0] ?? [];
    const entityEmbeddings = allEmbeddings.slice(
      1,
      1 + newEntityNames.length
    );
    const factEmbeddings = allEmbeddings.slice(1 + newEntityNames.length);

    // ── Step 5: Create and store the Passage object ──
    const passage: Passage = {
      id: passageId,
      text,
      embedding: passageEmbedding,
      triples,
      timestamp: Date.now(),
      salience,
      accessCount: 0,
      lastAccessed: 0,
    };
    this.passages.set(passageId, passage);
    this.chunkStore.insert(passageId, text, passage.embedding);

    // ── Step 6: Add passage node to graph ──
    this.graph.addNode(passageId, "passage");

    // ── Step 7: Store entity embeddings and build graph edges ──
    for (let i = 0; i < newEntityNames.length; i++) {
      const entity = newEntityNames[i]!;
      this.entityStore.insert(entity.id, entity.name, entityEmbeddings[i] ?? []);
    }

    for (let i = 0; i < newFactStrings.length; i++) {
      const fact = newFactStrings[i]!;
      this.factStore.insert(fact.id, fact.text, factEmbeddings[i] ?? []);
    }

    // ── Step 8: Add fact edges + passage edges to graph ──
    for (const triple of triples) {
      const subjectId = computeHashId(triple.subject, "entity-");
      const objectId = computeHashId(triple.object, "entity-");

      // Ensure entity nodes exist in the graph
      this.graph.addNode(subjectId, "entity");
      this.graph.addNode(objectId, "entity");

      // Fact edges: entity ↔ entity (bidirectional, weights accumulate)
      this.graph.addEdge(subjectId, objectId, 1.0);

      // Passage edges: passage → entity ("contains" relationship)
      this.graph.setEdge(passageId, subjectId, 1.0);
      this.graph.setEdge(passageId, objectId, 1.0);

      // Track entity → chunk mapping
      if (!this.entityToChunkIds.has(subjectId)) {
        this.entityToChunkIds.set(subjectId, new Set());
      }
      this.entityToChunkIds.get(subjectId)!.add(passageId);

      if (!this.entityToChunkIds.has(objectId)) {
        this.entityToChunkIds.set(objectId, new Set());
      }
      this.entityToChunkIds.get(objectId)!.add(passageId);
    }

    // ── Step 9: Add synonym edges ──
    // KNN on entity embeddings to find semantically similar entities
    // (e.g., "ML" ↔ "machine learning")
    if (newEntityNames.length > 0) {
      this.addSynonymEdges();
    }

    return passage;
  }

  // ══════════════════════════════════════════════
  // RETRIEVAL PIPELINE
  // ══════════════════════════════════════════════

  /**
   * Retrieve the most relevant stored passages for a query.
   *
   * This is the full HippoRAG2 retrieval pipeline:
   *   1. Embed the query
   *   2. Dense retrieval: find top-K triples by embedding similarity
   *   3. Recognition memory: LLM filters candidate triples
   *   4. Build PPR personalisation vector from filtered triples + DPR scores
   *   5. Run Personalized PageRank
   *   6. Extract and return top-K passage scores
   *
   * If no triples survive filtering, falls back to dense passage retrieval.
   *
   * @param query — the user's question / retrieval query
   * @param topK  — number of passages to return (default from config)
   * @returns array of Passage objects, most relevant first
   */
  async retrieve(query: string, topK?: number): Promise<Passage[]> {
    const k = topK ?? this.config.retrievalTopK;

    // If we have no indexed passages, return empty
    if (this.passages.size === 0) return [];

    // ── Step 1: Embed the query ──
    const queryEmbedding = await embedQuery(query);

    // ── Step 2: Get fact scores (dense triple retrieval) ──
    // Compute cosine similarity between query and all stored triple embeddings
    const factIds = this.factStore.getAllIds();
    const factScores: number[] = [];

    for (const factId of factIds) {
      const factEmb = this.factStore.getEmbedding(factId);
      if (factEmb) {
        factScores.push(cosineSimilarity(queryEmbedding, factEmb));
      } else {
        factScores.push(0);
      }
    }

    // Min-max normalize scores (matching HippoRAG Python)
    const normalizedFactScores = minMaxNormalize(factScores);

    // ── Step 3: Get top-K candidate triples ──
    // Create index-score pairs, sort descending, take top linkingTopK
    const candidateIndices = normalizedFactScores
      .map((score, idx) => ({ idx, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, this.config.linkingTopK);

    const candidateTriples: Triple[] = [];
    const candidateFactScores: number[] = [];

    for (const { idx, score } of candidateIndices) {
      const factId = factIds[idx];
      if (factId) {
        const triple = this.factIdToTriple.get(factId);
        if (triple) {
          candidateTriples.push(triple);
          candidateFactScores.push(score);
        }
      }
    }

    // ── Step 4: Recognition memory — LLM filters triples ──
    const filteredTriples = await filterTriples(query, candidateTriples);

    // ── Step 5: Build personalisation vector or fallback to DPR ──
    if (filteredTriples.length === 0) {
      // No relevant triples found — fallback to Dense Passage Retrieval
      return this.densePassageRetrieval(queryEmbedding, k);
    }

    // Build phrase (entity) weights from filtered triples
    // (matches graph_search_with_fact_entities in HippoRAG Python)
    const phraseWeights = new Map<string, number>();
    const phraseOccurrences = new Map<string, number>();

    for (let i = 0; i < filteredTriples.length; i++) {
      const triple = filteredTriples[i]!;
      // Find this triple's dense score (from the original candidate list)
      const originalIdx = candidateTriples.indexOf(triple);
      const factScore =
        originalIdx >= 0 ? (candidateFactScores[originalIdx] ?? 0) : 0;

      for (const entityName of [triple.subject, triple.object]) {
        const entityId = computeHashId(
          normalizeEntity(entityName),
          "entity-"
        );

        // Only include entities that exist in the graph
        if (!this.graph.hasNode(entityId)) continue;

        // Weight by inverse of chunk count (entities in fewer chunks are more specific)
        const chunkCount = this.entityToChunkIds.get(entityId)?.size ?? 1;
        const weightedScore = factScore / chunkCount;

        phraseWeights.set(
          entityId,
          (phraseWeights.get(entityId) ?? 0) + weightedScore
        );
        phraseOccurrences.set(
          entityId,
          (phraseOccurrences.get(entityId) ?? 0) + 1
        );
      }
    }

    // Average phrase weights by occurrence count (matching HippoRAG)
    for (const [entityId, totalWeight] of phraseWeights) {
      const occurrences = phraseOccurrences.get(entityId) ?? 1;
      phraseWeights.set(entityId, totalWeight / occurrences);
    }

    // Get dense passage retrieval scores for passage node weights
    const passageIds = Array.from(this.passages.keys());
    const passageScores: number[] = [];

    for (const pId of passageIds) {
      const pEmb = this.chunkStore.getEmbedding(pId);
      if (pEmb) {
        passageScores.push(cosineSimilarity(queryEmbedding, pEmb));
      } else {
        passageScores.push(0);
      }
    }

    const normalizedPassageScores = minMaxNormalize(passageScores);

    // Build the combined personalisation vector for PPR
    const resetProb = new Map<string, number>();

    // Add phrase weights
    for (const [entityId, weight] of phraseWeights) {
      resetProb.set(entityId, weight);
    }

    // Add passage weights (scaled by passageNodeWeight)
    for (let i = 0; i < passageIds.length; i++) {
      const pId = passageIds[i]!;
      const pScore = normalizedPassageScores[i] ?? 0;
      resetProb.set(pId, pScore * this.config.passageNodeWeight);
    }

    // ── Step 6: Run Personalized PageRank ──
    const pprScores = this.graph.personalizedPageRank(
      resetProb,
      this.config.damping
    );

    // ── Step 7: Extract passage scores and return top-K ──
    const rankedPassages = this.graph.getPassageScores(pprScores);

    const results: Passage[] = [];
    for (const { id } of rankedPassages.slice(0, k)) {
      const passage = this.passages.get(id);
      if (passage) {
        // Update access stats (for semantic forgetting recency boost)
        passage.accessCount++;
        passage.lastAccessed = Date.now();
        results.push(passage);
      }
    }

    return results;
  }

  // ══════════════════════════════════════════════
  // SEMANTIC FORGETTING
  // ══════════════════════════════════════════════

  /**
   * Prune old, low-salience passages when the store exceeds capacity.
   *
   * This implements the "semantic forgetting" from the paper:
   * age-weighted pruning of low-salience chunks. The forgetting score
   * balances three factors:
   *   - Salience: how important the LLM judged this memory
   *   - Age: how long ago it was stored
   *   - Recency: how recently it was accessed (retrieved)
   *
   * Passages with HIGH forgetting scores (unimportant + old + not recently
   * accessed) are removed first.
   *
   * When a passage is removed:
   *   - Its embedding is removed from the chunk store
   *   - Its graph node and all edges are removed
   *   - Entity nodes that no longer reference any passages are pruned
   *   - Facts unique to this passage are removed
   */
  forget(): void {
    if (this.passages.size <= this.config.maxPassages) return;

    const now = Date.now();
    const scored: { id: string; forgetScore: number }[] = [];

    for (const [id, passage] of this.passages) {
      // Age in hours since storage
      const ageHours = (now - passage.timestamp) / (1000 * 60 * 60);

      // Recency boost: higher if recently accessed
      const recencyBoost =
        passage.accessCount > 0
          ? 1 / (1 + (now - passage.lastAccessed) / (1000 * 60 * 60))
          : 0;

      // Forgetting score: higher = more forgettable
      // Low salience + old age = very forgettable
      // High salience + recent access = preserved
      const forgetScore =
        (1 - passage.salience) * Math.log(1 + ageHours) - recencyBoost;

      scored.push({ id, forgetScore });
    }

    // Sort: highest forgetting score first (most forgettable)
    scored.sort((a, b) => b.forgetScore - a.forgetScore);

    // Remove enough passages to get back under the limit
    const toRemove = this.passages.size - this.config.maxPassages;
    for (let i = 0; i < toRemove; i++) {
      const { id } = scored[i]!;
      this.removePassage(id);
    }
  }

  // ══════════════════════════════════════════════
  // PRIVATE HELPERS
  // ══════════════════════════════════════════════

  /**
   * Dense Passage Retrieval (DPR) — fallback when no triples match.
   *
   * Simple cosine similarity search over passage embeddings.
   * Used when the recognition memory filter removes all candidate triples,
   * meaning the knowledge graph can't help and we fall back to vanilla
   * vector search.
   */
  private densePassageRetrieval(
    queryEmbedding: number[],
    topK: number
  ): Passage[] {
    const results = this.chunkStore.similaritySearch(queryEmbedding, topK);
    const passages: Passage[] = [];

    for (const { id } of results) {
      const passage = this.passages.get(id);
      if (passage) {
        passage.accessCount++;
        passage.lastAccessed = Date.now();
        passages.push(passage);
      }
    }

    return passages;
  }

  /**
   * Add synonym edges between semantically similar entity nodes.
   *
   * This mirrors HippoRAG's `add_synonymy_edges()`. It runs KNN on all
   * entity embeddings and creates edges between entities whose cosine
   * similarity exceeds the threshold (default 0.8).
   *
   * Example: "ML" and "machine learning" would get a synonym edge.
   *
   * These edges are crucial for multi-hop reasoning — they let PPR
   * spread activation between different surface forms of the same concept.
   */
  private addSynonymEdges(): void {
    const entityIds = this.entityStore.getAllIds();
    if (entityIds.length < 2) return;

    const knnResults = this.entityStore.knnSearch(
      entityIds,
      entityIds,
      this.config.synonymyTopK,
      this.config.synonymyThreshold
    );

    for (const [queryId, neighbors] of knnResults) {
      // Skip very short entity names (likely noise)
      const entityContent = this.entityStore.getItem(queryId)?.content ?? "";
      if (entityContent.replace(/[^a-z0-9]/gi, "").length <= 2) continue;

      for (const { id: neighborId, score } of neighbors) {
        // Set (not add) the edge weight to the similarity score
        this.graph.setEdge(queryId, neighborId, score);
      }
    }
  }

  /**
   * Remove a passage and clean up all associated data.
   *
   * This is called during semantic forgetting. It removes:
   *   - The passage from the passages map
   *   - Its embedding from the chunk store
   *   - Its node from the knowledge graph (and all edges)
   *   - Entity→chunk mappings
   *   - Orphaned entities and facts
   */
  private removePassage(passageId: string): void {
    const passage = this.passages.get(passageId);
    if (!passage) return;

    // Remove from passage storage and chunk embedding store
    this.passages.delete(passageId);
    this.chunkStore.remove(passageId);

    // Remove from graph (removes all edges too)
    this.graph.removeNode(passageId);

    // Clean up entity→chunk mappings
    for (const triple of passage.triples) {
      const subjectId = computeHashId(triple.subject, "entity-");
      const objectId = computeHashId(triple.object, "entity-");

      // Remove this passage from entity's chunk set
      this.entityToChunkIds.get(subjectId)?.delete(passageId);
      this.entityToChunkIds.get(objectId)?.delete(passageId);

      // If an entity no longer references any passages, remove it entirely
      for (const entityId of [subjectId, objectId]) {
        const chunkIds = this.entityToChunkIds.get(entityId);
        if (chunkIds && chunkIds.size === 0) {
          this.entityToChunkIds.delete(entityId);
          this.entityStore.remove(entityId);
          this.graph.removeNode(entityId);
        }
      }

      // Remove the fact embedding if no other passage uses it
      const factStr = tripleToString(triple);
      const factId = computeHashId(factStr, "fact-");
      // Check if any other passage has the same triple
      let stillUsed = false;
      for (const [, otherPassage] of this.passages) {
        if (otherPassage.triples.some(
          (t) => tripleToString(t) === factStr
        )) {
          stillUsed = true;
          break;
        }
      }
      if (!stillUsed) {
        this.factStore.remove(factId);
        this.factIdToTriple.delete(factId);
      }
    }
  }

  // ══════════════════════════════════════════════
  // DEBUG / INSPECTION
  // ══════════════════════════════════════════════

  /**
   * Get stats about the current state of the HippoRAG index.
   * Useful for debugging and monitoring.
   */
  getStats(): {
    passages: number;
    entities: number;
    facts: number;
    graphNodes: number;
  } {
    return {
      passages: this.passages.size,
      entities: this.entityStore.size,
      facts: this.factStore.size,
      graphNodes: this.graph.nodeCount,
    };
  }
}
