/**
 * graph.ts — LangGraph StateGraph definition.
 *
 *   START → respond → memorize → END
 *
 * The respond node has a `remember` tool with optional tag filters.
 * The memorize node handles pressure-based indexing into HippoRAG.
 */

import { StateGraph, START, END } from "@langchain/langgraph";
import { BrainyState } from "./state.ts";
import { respondNode } from "./nodes/respond.ts";
import { memorizeNode } from "./nodes/memorize.ts";

export function buildGraph() {
  const graph = new StateGraph(BrainyState)
    .addNode("respond", respondNode)
    .addNode("memorize", memorizeNode)

    .addEdge(START, "respond")
    .addEdge("respond", "memorize")
    .addEdge("memorize", END)

    .compile();

  return graph;
}
