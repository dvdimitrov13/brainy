/**
 * graph.ts — LangGraph StateGraph definition.
 *
 * The graph has a conditional loop for memory pressure:
 *
 *   START → respond → memorize → END        (normal flow)
 *   START → respond → memorize → respond     (pressure: must write notes)
 *
 * When the memorize node detects buffer pressure, it sets mustWriteNotes
 * and the graph loops back to respond. The respond node forces the agent
 * to call write_notes, then proceeds to memorize again which clears the
 * pressure and routes to END.
 */

import { StateGraph, START, END } from "@langchain/langgraph";
import { BrainyState } from "./state.ts";
import { respondNode } from "./nodes/respond.ts";
import { memorizeNode } from "./nodes/memorize.ts";

/**
 * Build and compile the Brainy conversation graph.
 */
export function buildGraph() {
  const graph = new StateGraph(BrainyState)
    .addNode("respond", respondNode)
    .addNode("memorize", memorizeNode)

    .addEdge(START, "respond")
    .addEdge("respond", "memorize")

    // Conditional edge: memorize → respond (if pressure) or memorize → END
    .addConditionalEdges("memorize", (state) => {
      return state.mustWriteNotes ? "respond" : "__end__";
    })

    .compile();

  return graph;
}
