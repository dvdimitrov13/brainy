/**
 * graph.ts — LangGraph StateGraph definition.
 *
 *   START → retrieve → respond → memorize → END
 *
 * retrieve: auto-recognize associations every turn (Haiku contextualization)
 * respond: LLM with 4 tools (recognize, recall, remember, explore_topics)
 * memorize: pressure-based indexing into HippoRAG
 */

import { StateGraph, START, END } from "@langchain/langgraph";
import { BrainyState } from "./state.ts";
import { retrieveNode } from "./nodes/retrieve.ts";
import { respondNode } from "./nodes/respond.ts";
import { memorizeNode } from "./nodes/memorize.ts";

export function buildGraph() {
  const graph = new StateGraph(BrainyState)
    .addNode("retrieve", retrieveNode)
    .addNode("respond", respondNode)
    .addNode("memorize", memorizeNode)

    .addEdge(START, "retrieve")
    .addEdge("retrieve", "respond")
    .addEdge("respond", "memorize")
    .addEdge("memorize", END)

    .compile();

  return graph;
}
