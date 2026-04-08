/**
 * knowledge-graph.ts — Graph data structure + Personalized PageRank (PPR).
 *
 * This is the core of HippoRAG2. It implements:
 *   1. A weighted undirected graph with two node types (entity + passage)
 *   2. Three edge types (fact, passage, synonym)
 *   3. Personalized PageRank via power iteration
 *
 * In the Python HippoRAG codebase, this role is played by `igraph.Graph`
 * and its `.personalized_pagerank()` method. Since igraph doesn't exist
 * in JS, we implement PPR from scratch using power iteration (which is
 * what igraph's "prpack" solver does under the hood).
 *
 * TS notes for Python devs:
 *   - `Map<string, Map<string, number>>` is like a nested dict:
 *     `dict[str, dict[str, float]]` — an adjacency list where
 *     graph["alice"]["bob"] = 0.5 means there's an edge with weight 0.5.
 *   - We use `Map` instead of plain objects because Map supports
 *     efficient iteration and has a `.size` property.
 */

import type { GraphNode } from "./types.ts";

export class KnowledgeGraph {
  // ──────────────────────────────────────────────
  // Node storage
  // ──────────────────────────────────────────────

  /** Map from node ID → metadata (type + matrix index).
   *  Python equivalent: dict[str, GraphNode] */
  private nodes: Map<string, GraphNode> = new Map();

  /** Ordered list of node IDs — index position matches the matrix row/col.
   *  Python equivalent: list[str] */
  private nodeIds: string[] = [];

  // ──────────────────────────────────────────────
  // Edge storage (adjacency list)
  // ──────────────────────────────────────────────

  /** Weighted adjacency list: nodeId → { neighborId → weight }.
   *  Undirected: if A→B exists, B→A also exists.
   *  Python equivalent: defaultdict(lambda: defaultdict(float)) */
  private adjacency: Map<string, Map<string, number>> = new Map();

  // ──────────────────────────────────────────────
  // Node operations
  // ──────────────────────────────────────────────

  /**
   * Add a node to the graph.
   *
   * If the node already exists, this is a no-op.
   *
   * @param id   — unique node ID (e.g. "entity-a1b2c3d4" or "passage-xyz")
   * @param type — 'entity' (phrase node) or 'passage'
   */
  addNode(id: string, type: "entity" | "passage"): void {
    if (this.nodes.has(id)) return; // already exists

    const index = this.nodeIds.length;
    this.nodes.set(id, { type, index });
    this.nodeIds.push(id);
    // Initialise empty adjacency set for this node
    this.adjacency.set(id, new Map());
  }

  /**
   * Check if a node exists in the graph.
   */
  hasNode(id: string): boolean {
    return this.nodes.has(id);
  }

  /**
   * Get metadata for a node.
   */
  getNode(id: string): GraphNode | undefined {
    return this.nodes.get(id);
  }

  /**
   * Get the total number of nodes.
   */
  get nodeCount(): number {
    return this.nodes.size;
  }

  /**
   * Get all node IDs of a specific type.
   */
  getNodeIdsByType(type: "entity" | "passage"): string[] {
    return this.nodeIds.filter((id) => this.nodes.get(id)?.type === type);
  }

  // ──────────────────────────────────────────────
  // Edge operations
  // ──────────────────────────────────────────────

  /**
   * Add an undirected weighted edge between two nodes.
   *
   * If the edge already exists, the weight is ADDED to the existing weight
   * (accumulate co-occurrence counts, like HippoRAG's fact edges).
   *
   * Both nodes must already exist in the graph.
   *
   * @param source — first node ID
   * @param target — second node ID
   * @param weight — edge weight (default 1.0)
   */
  addEdge(source: string, target: string, weight = 1.0): void {
    // Ensure both adjacency maps exist
    if (!this.adjacency.has(source)) this.adjacency.set(source, new Map());
    if (!this.adjacency.has(target)) this.adjacency.set(target, new Map());

    const sourceAdj = this.adjacency.get(source)!;
    const targetAdj = this.adjacency.get(target)!;

    // Accumulate weight (HippoRAG adds co-occurrence counts)
    sourceAdj.set(target, (sourceAdj.get(target) ?? 0) + weight);
    targetAdj.set(source, (targetAdj.get(source) ?? 0) + weight);
  }

  /**
   * Set an edge weight directly (no accumulation).
   *
   * Used for synonym edges and passage→entity edges where we want
   * to set the weight to a specific value, not accumulate.
   */
  setEdge(source: string, target: string, weight: number): void {
    if (!this.adjacency.has(source)) this.adjacency.set(source, new Map());
    if (!this.adjacency.has(target)) this.adjacency.set(target, new Map());

    this.adjacency.get(source)!.set(target, weight);
    this.adjacency.get(target)!.set(source, weight);
  }

  /**
   * Get the weight of an edge between two nodes.
   *
   * @returns the weight, or 0 if no edge exists
   */
  getEdgeWeight(source: string, target: string): number {
    return this.adjacency.get(source)?.get(target) ?? 0;
  }

  /**
   * Get all neighbors of a node and their edge weights.
   */
  getNeighbors(nodeId: string): Map<string, number> {
    return this.adjacency.get(nodeId) ?? new Map();
  }

  // ──────────────────────────────────────────────
  // Node removal (for semantic forgetting)
  // ──────────────────────────────────────────────

  /**
   * Remove a node and all its edges from the graph.
   *
   * This is used during semantic forgetting to prune old passage nodes.
   * After removal, we DON'T re-index (the index array has a gap) — PPR
   * just skips removed indices. This is simpler than rebuilding indices
   * and matches how igraph handles vertex deletion.
   *
   * @param id — node ID to remove
   */
  removeNode(id: string): void {
    const node = this.nodes.get(id);
    if (!node) return;

    // Remove all edges TO this node from its neighbors
    const neighbors = this.adjacency.get(id);
    if (neighbors) {
      for (const [neighborId] of neighbors) {
        this.adjacency.get(neighborId)?.delete(id);
      }
    }

    // Remove the node's own adjacency entry
    this.adjacency.delete(id);

    // Remove from node map (but NOT from nodeIds — we leave a gap)
    this.nodes.delete(id);
  }

  // ──────────────────────────────────────────────
  // Personalized PageRank (PPR)
  // ──────────────────────────────────────────────

  /**
   * Run Personalized PageRank over the graph.
   *
   * This is the heart of HippoRAG2's retrieval. It takes a personalisation
   * vector (the "reset probabilities") and computes a stationary distribution
   * where nodes connected to high-seed entities receive high scores.
   *
   * Algorithm (power iteration):
   *   1. Build column-normalised transition matrix M from adjacency weights
   *   2. Normalise personalisation vector p = resetProb / sum(resetProb)
   *   3. Init score vector x = p
   *   4. Iterate: x_new = (1 - damping) * p + damping * M @ x
   *   5. Until convergence: max(|x_new - x_old|) < epsilon or maxIter
   *
   * This matches igraph's `personalized_pagerank(implementation='prpack')`.
   *
   * The damping factor (default 0.5 in HippoRAG2) controls how far
   * activation spreads. Lower damping = more teleportation back to seeds
   * = more local results. Higher damping = more graph exploration.
   *
   * @param resetProb — Map from nodeId → reset probability.
   *                    Only nodes with non-zero values are "seed" nodes.
   *                    The map does NOT need to cover all nodes — missing
   *                    nodes get 0.
   * @param damping   — probability of following an edge (default 0.5)
   * @returns Map from nodeId → PPR score (higher = more relevant)
   */
  personalizedPageRank(
    resetProb: Map<string, number>,
    damping = 0.5
  ): Map<string, number> {
    const n = this.nodeIds.length;
    if (n === 0) return new Map();

    // ── Step 1: Build personalisation vector ──
    // Convert the sparse resetProb map into a dense array aligned with nodeIds
    const p = new Float64Array(n);
    let pSum = 0;
    for (let i = 0; i < n; i++) {
      const nodeId = this.nodeIds[i]!;
      // Skip removed nodes
      if (!this.nodes.has(nodeId)) continue;
      const val = resetProb.get(nodeId) ?? 0;
      // Clamp negatives and NaN to 0 (matches HippoRAG Python code)
      p[i] = isNaN(val) || val < 0 ? 0 : val;
      pSum += p[i]!;
    }

    // If no seeds, return all zeros
    if (pSum === 0) {
      return new Map(this.nodeIds.map((id) => [id, 0]));
    }

    // Normalise so p sums to 1 (probability distribution)
    for (let i = 0; i < n; i++) {
      p[i]! /= pSum;
    }

    // ── Step 2: Build column-normalised transition matrix ──
    // For sparse graphs we store it as an array of { targetIdx, weight }
    // per source node, where weights are normalised per source.
    //
    // M[j][i] = weight(i→j) / sum_k(weight(i→k))
    // (column-normalised: columns sum to 1)
    //
    // We store it row-wise for iteration efficiency.
    const outWeights: Float64Array = new Float64Array(n); // sum of outgoing weights per node
    for (let i = 0; i < n; i++) {
      const nodeId = this.nodeIds[i]!;
      if (!this.nodes.has(nodeId)) continue;
      const neighbors = this.adjacency.get(nodeId);
      if (!neighbors) continue;
      let sum = 0;
      for (const [, w] of neighbors) sum += w;
      outWeights[i] = sum;
    }

    // ── Step 3: Power iteration ──
    let x = new Float64Array(p); // x = p (initial scores)
    const maxIter = 100;
    const epsilon = 1e-6;

    for (let iter = 0; iter < maxIter; iter++) {
      const xNew = new Float64Array(n);

      // Teleportation component: (1 - damping) * p
      for (let i = 0; i < n; i++) {
        xNew[i] = (1 - damping) * p[i]!;
      }

      // Transition component: damping * M @ x
      // For each node i with score x[i], distribute damping * x[i] * w(i→j) / outWeight[i]
      // to each neighbor j
      for (let i = 0; i < n; i++) {
        const nodeId = this.nodeIds[i]!;
        if (!this.nodes.has(nodeId)) continue;
        if (outWeights[i] === 0) continue; // dangling node: its score just teleports

        const score = x[i]!;
        const neighbors = this.adjacency.get(nodeId);
        if (!neighbors) continue;

        for (const [neighborId, weight] of neighbors) {
          const neighborNode = this.nodes.get(neighborId);
          if (!neighborNode) continue;
          const j = neighborNode.index;
          // Add damping * x[i] * (weight / outWeight[i])
          xNew[j]! += damping * score * (weight / outWeights[i]!);
        }
      }

      // Check convergence: max absolute difference
      let maxDiff = 0;
      for (let i = 0; i < n; i++) {
        const diff = Math.abs(xNew[i]! - x[i]!);
        if (diff > maxDiff) maxDiff = diff;
      }

      x = xNew;
      if (maxDiff < epsilon) break;
    }

    // ── Step 4: Build result map ──
    const result = new Map<string, number>();
    for (let i = 0; i < n; i++) {
      const nodeId = this.nodeIds[i]!;
      if (this.nodes.has(nodeId)) {
        result.set(nodeId, x[i]!);
      }
    }
    return result;
  }

  /**
   * Extract only passage node scores from PPR results.
   *
   * After PPR, we only care about passage scores — entity scores were
   * just conduits for spreading activation.
   *
   * @param pprScores — full PPR score map (all nodes)
   * @returns array of { id, score } for passage nodes, sorted descending
   */
  getPassageScores(
    pprScores: Map<string, number>
  ): { id: string; score: number }[] {
    const passageScores: { id: string; score: number }[] = [];

    for (const [nodeId, score] of pprScores) {
      const node = this.nodes.get(nodeId);
      if (node?.type === "passage") {
        passageScores.push({ id: nodeId, score });
      }
    }

    // Sort descending by score
    passageScores.sort((a, b) => b.score - a.score);
    return passageScores;
  }
}
