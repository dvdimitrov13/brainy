/**
 * embedding-store.ts — In-memory store for text embeddings.
 *
 * This mirrors HippoRAG's `EmbeddingStore` class. It holds:
 *   - The raw text content (entity name, passage text, or triple string)
 *   - The embedding vector (from Voyage 3.5)
 *
 * It supports similarity search (cosine) and KNN operations needed
 * for the retrieval pipeline and synonym detection.
 *
 * Python analogy: this is like a dict[str, { content: str, embedding: list[float] }]
 * with some search methods bolted on.
 *
 * TS note for Python devs:
 *   - `Map<K, V>` is like Python's `dict[K, V]` but with methods like
 *     `.get()`, `.set()`, `.has()`, `.delete()`, `.size` (not `len()`).
 *   - `private` means the field is only accessible inside this class
 *     (Python uses `_` prefix by convention, TS enforces it at compile time).
 */

import { cosineSimilarity } from "../utils.ts";

/** A single stored item: its text and embedding vector */
interface StoredItem {
  id: string;
  content: string;
  embedding: number[];
}

export class EmbeddingStore {
  /** Internal storage: id → { content, embedding } */
  private items: Map<string, StoredItem> = new Map();

  /**
   * Store a single text and its embedding.
   *
   * If an item with this ID already exists, it's overwritten.
   *
   * @param id        — unique identifier (e.g. "entity-a1b2c3d4")
   * @param content   — the raw text
   * @param embedding — the Voyage 3.5 embedding vector
   */
  insert(id: string, content: string, embedding: number[]): void {
    this.items.set(id, { id, content, embedding });
  }

  /**
   * Bulk insert multiple items at once.
   *
   * @param entries — array of { id, content, embedding } objects
   */
  batchInsert(entries: { id: string; content: string; embedding: number[] }[]): void {
    for (const entry of entries) {
      this.items.set(entry.id, entry);
    }
  }

  /**
   * Get the embedding vector for a single ID.
   *
   * @returns the embedding, or undefined if not found
   */
  getEmbedding(id: string): number[] | undefined {
    return this.items.get(id)?.embedding;
  }

  /**
   * Get the full stored item (id + content + embedding) for a single ID.
   */
  getItem(id: string): StoredItem | undefined {
    return this.items.get(id);
  }

  /**
   * Get all stored IDs.
   *
   * TS note: `IterableIterator<string>` is like Python's `dict.keys()`.
   * You can use `Array.from(store.getAllIds())` to convert to an array,
   * or iterate with `for (const id of store.getAllIds())`.
   */
  getAllIds(): string[] {
    return Array.from(this.items.keys());
  }

  /**
   * Get all embeddings as a 2D array, in the same order as getAllIds().
   *
   * Used for batch cosine similarity computation (like numpy dot product).
   */
  getAllEmbeddings(): number[][] {
    return Array.from(this.items.values()).map((item) => item.embedding);
  }

  /**
   * Get all embeddings keyed by ID.
   */
  getEmbeddingsById(ids: string[]): Map<string, number[]> {
    const result = new Map<string, number[]>();
    for (const id of ids) {
      const emb = this.items.get(id)?.embedding;
      if (emb) result.set(id, emb);
    }
    return result;
  }

  /**
   * Search for the most similar items to a query embedding.
   *
   * This is a linear scan (O(n) in the number of stored items).
   * Fine for < 10K items; for production scale you'd use an ANN index.
   *
   * @param queryEmbedding — the query vector
   * @param topK           — number of results to return
   * @returns array of { id, content, score } sorted by similarity (descending)
   */
  similaritySearch(
    queryEmbedding: number[],
    topK: number
  ): { id: string; content: string; score: number }[] {
    const scores: { id: string; content: string; score: number }[] = [];

    for (const item of this.items.values()) {
      const score = cosineSimilarity(queryEmbedding, item.embedding);
      scores.push({ id: item.id, content: item.content, score });
    }

    // Sort descending by score and take top-K
    // Python equivalent: sorted(scores, key=lambda x: x['score'], reverse=True)[:topK]
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
  }

  /**
   * KNN search for synonym detection.
   *
   * For each query entity, find the top-K most similar entities
   * that are above the similarity threshold.
   *
   * This is used to create synonym edges in the knowledge graph
   * (e.g., "ML" and "machine learning" should be linked).
   *
   * @param queryIds    — IDs of entities to find synonyms for
   * @param keyIds      — IDs of all candidate entities
   * @param topK        — max synonyms per entity
   * @param threshold   — minimum cosine similarity to be considered a synonym
   * @returns Map from queryId → array of { id, score } (the nearest neighbors)
   */
  knnSearch(
    queryIds: string[],
    keyIds: string[],
    topK: number,
    threshold: number
  ): Map<string, { id: string; score: number }[]> {
    const result = new Map<string, { id: string; score: number }[]>();

    // Pre-fetch key embeddings for efficiency
    const keyEmbeddings: { id: string; embedding: number[] }[] = [];
    for (const keyId of keyIds) {
      const emb = this.items.get(keyId)?.embedding;
      if (emb) keyEmbeddings.push({ id: keyId, embedding: emb });
    }

    for (const queryId of queryIds) {
      const queryEmb = this.items.get(queryId)?.embedding;
      if (!queryEmb) continue;

      const neighbors: { id: string; score: number }[] = [];

      for (const key of keyEmbeddings) {
        // Skip self-comparisons
        if (key.id === queryId) continue;

        const score = cosineSimilarity(queryEmb, key.embedding);
        if (score >= threshold) {
          neighbors.push({ id: key.id, score });
        }
      }

      // Sort by score descending and cap at topK
      neighbors.sort((a, b) => b.score - a.score);
      result.set(queryId, neighbors.slice(0, topK));
    }

    return result;
  }

  /**
   * Remove an item by ID.
   *
   * Used during semantic forgetting to clean up orphaned entities.
   *
   * @returns true if the item existed and was removed
   */
  remove(id: string): boolean {
    return this.items.delete(id);
  }

  /**
   * How many items are stored.
   *
   * TS note: `get` before a method name makes it a "getter" property,
   * so you can access it as `store.size` (no parentheses) like Python's
   * `@property` decorator.
   */
  get size(): number {
    return this.items.size;
  }

  /**
   * Check if an item exists.
   */
  has(id: string): boolean {
    return this.items.has(id);
  }
}
