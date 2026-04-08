/**
 * types.ts — All TypeScript types for the HippoRAG2 module.
 *
 * Python analogy: these are like @dataclass or TypedDict definitions.
 * In TypeScript, `interface` defines a "shape" — an object must have
 * these fields with these types. Unlike Python dataclasses, interfaces
 * have ZERO runtime cost (they're erased during compilation).
 *
 * `export` makes the type available to other files via:
 *   import type { Triple } from "./types.ts";
 *
 * The `type` keyword in the import tells TS "this is only used for
 * type-checking, not at runtime" — a best practice with verbatimModuleSyntax.
 */

// ──────────────────────────────────────────────
// Knowledge Graph Types
// ──────────────────────────────────────────────

/**
 * A (subject, predicate, object) triple extracted from text via OpenIE.
 *
 * Example: { subject: "alice", predicate: "works_at", object: "google" }
 *
 * All strings are normalized to lowercase for consistency.
 */
export interface Triple {
  subject: string;
  predicate: string;
  object: string;
}

/**
 * A stored passage (conversation chunk) in the HippoRAG index.
 *
 * Each passage corresponds to one conversation exchange (user + assistant).
 * The embedding is computed via Voyage 3.5 and stored alongside the text.
 */
export interface Passage {
  /** Unique content-based ID (e.g. "passage-a1b2c3d4") */
  id: string;
  /** The raw conversation text */
  text: string;
  /** Voyage 3.5 embedding vector */
  embedding: number[];
  /** The triples extracted from this passage */
  triples: Triple[];
  /** Unix timestamp (Date.now()) when the passage was indexed */
  timestamp: number;
  /** LLM-judged importance score, 0-1 (used for semantic forgetting) */
  salience: number;
  /** How many times this passage has been retrieved */
  accessCount: number;
  /** Timestamp of the last time this passage was retrieved */
  lastAccessed: number;
}

/**
 * Metadata for a node in the knowledge graph.
 *
 * The graph has two types of nodes:
 *   - 'entity' (phrase nodes): extracted entities like "alice", "machine learning"
 *   - 'passage': the stored conversation chunks
 *
 * `index` is the node's position in the ordered node list, used for
 * matrix operations during PPR (Personalized PageRank).
 */
export interface GraphNode {
  type: "entity" | "passage";
  index: number;
}

/**
 * Configuration for the HippoRAG2 system.
 *
 * These defaults match the HippoRAG2 paper's recommended values.
 * You can override any of them when constructing the HippoRAG instance.
 */
export interface HippoRAGConfig {
  /** PPR damping factor — probability of following an edge vs teleporting.
   *  0.5 is the HippoRAG2 default (igraph convention). */
  damping: number;

  /** Weight scalar for passage nodes in the PPR personalization vector.
   *  Low value (0.05) so phrase nodes dominate the seed signal. */
  passageNodeWeight: number;

  /** Number of top triples to retrieve for recognition memory filtering. */
  linkingTopK: number;

  /** Cosine similarity threshold for creating synonym edges between entities. */
  synonymyThreshold: number;

  /** KNN k for synonym edge search. */
  synonymyTopK: number;

  /** Maximum stored passages before semantic forgetting kicks in. */
  maxPassages: number;

  /** Default number of passages to return per retrieval query. */
  retrievalTopK: number;
}

/**
 * Default configuration values matching the HippoRAG2 paper.
 */
export const DEFAULT_CONFIG: HippoRAGConfig = {
  damping: 0.5,
  passageNodeWeight: 0.05,
  linkingTopK: 5,
  synonymyThreshold: 0.8,
  synonymyTopK: 10,
  maxPassages: 200,
  retrievalTopK: 3,
};
