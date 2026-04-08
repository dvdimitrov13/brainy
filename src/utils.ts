/**
 * utils.ts — Small helper functions used across the project.
 *
 * Keeping these in one file avoids scattering tiny utilities everywhere.
 * Each function is pure (no side effects) and well-typed.
 */

/**
 * Cosine similarity between two vectors.
 *
 * Formula:  cos(a,b) = (a · b) / (‖a‖ × ‖b‖)
 *
 * Returns a value between -1 and 1 (1 = identical direction, 0 = orthogonal).
 * If either vector is zero-length, returns 0.
 *
 * Python equivalent:
 *   from numpy import dot
 *   from numpy.linalg import norm
 *   cos_sim = dot(a, b) / (norm(a) * norm(b))
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

/**
 * Min-max normalise an array of numbers to [0, 1].
 *
 * If all values are the same (max === min), returns an array of zeros.
 * This matches the HippoRAG Python codebase's `min_max_normalize`.
 *
 * @param values — raw scores
 * @returns        normalised scores in [0, 1]
 */
export function minMaxNormalize(values: number[]): number[] {
  if (values.length === 0) return [];

  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const range = max - min;
  if (range === 0) return values.map(() => 0);

  return values.map((v) => (v - min) / range);
}

/**
 * Extract JSON from an LLM response that might be wrapped in markdown
 * code fences (```json ... ```).
 *
 * LLMs commonly return JSON inside code fences. This function strips
 * the fences so we can JSON.parse the raw content.
 *
 * @param text — raw LLM response text
 * @returns      the inner JSON string (still needs JSON.parse)
 */
export function extractJsonFromResponse(text: string): string {
  // Try markdown code fence first:  ```json\n{...}\n```
  const fenceMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (fenceMatch) return fenceMatch[1]!.trim();

  // Fallback: find the first { ... } block
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) return jsonMatch[0]!;

  // Last resort: return as-is and let JSON.parse throw
  return text;
}

/**
 * Generate a deterministic hash ID for a string, with an optional prefix.
 *
 * This mirrors HippoRAG's `compute_mdhash_id` — it creates stable,
 * content-based IDs so the same entity always maps to the same node ID.
 *
 * Uses a simple FNV-1a hash (fast, good distribution, no crypto needed).
 *
 * @param content — the string to hash
 * @param prefix  — optional prefix (e.g. "entity-", "passage-")
 * @returns         a string like "entity-a1b2c3d4"
 */
export function computeHashId(content: string, prefix = ""): string {
  // FNV-1a 32-bit hash
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < content.length; i++) {
    hash ^= content.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  // Convert to unsigned 32-bit hex string
  const hex = (hash >>> 0).toString(16).padStart(8, "0");
  return `${prefix}${hex}`;
}

/**
 * Normalise an entity name: lowercase, trim whitespace.
 *
 * This is a simple normalisation to reduce duplicates when the LLM
 * extracts "John" in one turn and "john" in another.
 */
export function normalizeEntity(name: string): string {
  return name.toLowerCase().trim();
}
