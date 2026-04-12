/**
 * graph.ts — LangGraph StateGraph definition.
 *
 * The graph is a two-node pipeline:
 *
 *   START → respond → memorize → END
 *
 * The respond node has access to two tools (recognize + recall) that
 * it calls as needed. No automatic memory injection — the agent
 * decides when to search its memory.
 *
 * TS note for Python devs:
 *   - `StateGraph(StateAnnotation)` is like `StateGraph(TypedDict)` in Python
 *   - `.addNode("name", func)` registers a node function
 *   - `.addEdge(A, B)` means "after A finishes, run B"
 *   - `START` and `END` are special constants for the graph entry/exit
 *   - `.compile()` returns an executable graph (like building a pipeline)
 */

import { StateGraph, START, END } from "@langchain/langgraph";
import { BrainyState } from "./state.ts";
import { respondNode } from "./nodes/respond.ts";
import { memorizeNode } from "./nodes/memorize.ts";

/**
 * Build and compile the Brainy conversation graph.
 *
 * @returns a compiled graph that can be invoked with `graph.invoke(state)`
 */
export function buildGraph() {
  const graph = new StateGraph(BrainyState)
    .addNode("respond", respondNode)
    .addNode("memorize", memorizeNode)

    // START → respond (with tool calls) → memorize → END
    .addEdge(START, "respond")
    .addEdge("respond", "memorize")
    .addEdge("memorize", END)

    .compile();

  return graph;
}
