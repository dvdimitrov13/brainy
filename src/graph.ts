/**
 * graph.ts — LangGraph StateGraph definition.
 *
 * This wires the three nodes into a linear pipeline:
 *
 *   START → retrieve → respond → memorize → END
 *
 * The graph is "compiled" into an executable that can be invoked
 * with `graph.invoke(state)`. Each invocation processes one
 * conversation turn.
 *
 * Why no conditional edges?
 *   This agent follows the same flow every turn — retrieve memories,
 *   respond, store new memories. There's no tool-use loop or branching.
 *   A ReAct-style agent would need conditional edges for the "should I
 *   use a tool?" decision, but our agent's "tools" are its memory stores,
 *   and they're always consulted.
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
import { retrieveNode } from "./nodes/retrieve.ts";
import { respondNode } from "./nodes/respond.ts";
import { memorizeNode } from "./nodes/memorize.ts";

/**
 * Build and compile the Brainy conversation graph.
 *
 * @returns a compiled graph that can be invoked with `graph.invoke(state)`
 */
export function buildGraph() {
  const graph = new StateGraph(BrainyState)
    // Register the three pipeline nodes
    .addNode("retrieve", retrieveNode)
    .addNode("respond", respondNode)
    .addNode("memorize", memorizeNode)

    // Wire them in sequence: START → retrieve → respond → memorize → END
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "respond")
    .addEdge("respond", "memorize")
    .addEdge("memorize", END)

    // Compile into an executable graph
    .compile();

  return graph;
}
