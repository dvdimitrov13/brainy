/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * This node assembles the prompt with memory context and calls the LLM.
 * The agent sees two types of memory:
 *
 *   1. Conversation buffer — real recent turns (within current session)
 *   2. Retrieved triples — lightweight entity associations from HippoRAG
 *      that surface automatically every turn
 *
 * The agent also has access to a "recall" tool. When it sees relevant
 * triples and wants the actual passage content, it calls the tool to
 * trigger full PPR retrieval over the knowledge graph. This mirrors
 * how memory works: associations surface automatically, but recalling
 * the full context takes deliberate effort.
 *
 * Graph position: START → retrieve → [respond] → memorize → END
 */

import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { BrainyState } from "../state.ts";
import { llm } from "../llm.ts";
import { hipporag } from "../singletons.ts";
import { chunkRerankPack } from "../chunking.ts";
import type { Triple } from "../hipporag/types.ts";

/**
 * The recall tool definition for the LLM.
 *
 * This tells the LLM it has a tool it can call to retrieve full memory
 * passages from long-term memory. The LLM decides when to use it based
 * on the triples it sees in the prompt.
 */
const RECALL_TOOL = {
  type: "function" as const,
  function: {
    name: "recall_memory",
    description:
      "Retrieve full conversation passages from long-term memory. " +
      "Use this when you see relevant memory associations (triples) and " +
      "need the actual conversation content to answer accurately. " +
      "Pass the user's message as the query.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description: "The query to search long-term memory for. Usually the user's question or a focused sub-question.",
        },
      },
      required: ["query"],
    },
  },
};

/**
 * Parse triples string back into Triple objects for PPR seeding.
 * Format: "1. (subject, predicate, object)"
 */
function parseTriplesFromContext(triplesStr: string): Triple[] {
  if (!triplesStr) return [];
  const triples: Triple[] = [];
  for (const line of triplesStr.split("\n")) {
    const match = line.match(/\(([^,]+),\s*([^,]+),\s*([^)]+)\)/);
    if (match) {
      triples.push({
        subject: match[1]!.trim(),
        predicate: match[2]!.trim(),
        object: match[3]!.trim(),
      });
    }
  }
  return triples;
}

/**
 * Generate an AI response using memory-augmented context.
 *
 * Implements a tool-calling loop:
 *   1. Call LLM with triples context + recall tool
 *   2. If LLM calls recall_memory → execute retrieval, feed results back
 *   3. Repeat until LLM gives a final text response
 */
export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  // ── Build the memory block for the system prompt ──
  const memoryParts: string[] = [];

  if (state.conversationBuffer) {
    memoryParts.push(
      `Conversation so far:\n${state.conversationBuffer}`
    );
  }

  if (state.retrievedTriples) {
    memoryParts.push(
      `Memory associations (entity relationships from past conversations):\n${state.retrievedTriples}`
    );
  }

  const memoryBlock = memoryParts.join("\n\n");

  const systemPrompt = `You are a helpful, friendly assistant with a long-term memory system.

You have two types of memory available:
1. Recent conversation context (shown below if any)
2. Memory associations — entity relationships from past conversations that may be relevant

If you see relevant memory associations and need to recall the actual conversation details, use the recall_memory tool. This searches your long-term memory for full conversation passages related to your query.

Do NOT explicitly mention your memory system. Just respond naturally as if you remember.

${memoryBlock ? `--- Your Memories ---\n${memoryBlock}\n--- End Memories ---` : "(No memories yet — this is the start of the conversation.)"}`;

  // ── Tool-calling loop ──
  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  // Parse the triples so we can seed PPR when the tool is called
  const triples = parseTriplesFromContext(state.retrievedTriples);

  const maxToolCalls = 3; // Safety limit

  for (let i = 0; i < maxToolCalls; i++) {
    const response = await llm.invoke(messages, {
      tools: [RECALL_TOOL],
    });

    // Check if the LLM wants to call the recall tool
    const toolCalls = response.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      // No tool call — this is the final response
      const aiResponse =
        typeof response.content === "string"
          ? response.content
          : (response.content as Array<{ type: string; text?: string }>)
              .filter((block) => block.type === "text")
              .map((block) => block.text ?? "")
              .join("");

      return { aiResponse };
    }

    // Execute tool calls
    messages.push(response);

    for (const toolCall of toolCalls) {
      if (toolCall.name === "recall_memory") {
        const query = (toolCall.args as { query: string }).query;

        // Phase 2: full PPR retrieval using the triples from Phase 1
        const passages = await hipporag.retrievePassages(
          query,
          triples
        );

        // Chunk, rerank, and pack within 1024 token budget
        const passageText = await chunkRerankPack(
          query,
          passages.map((p) => p.text)
        );

        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? `call_${i}`,
            content: passageText,
          })
        );
      }
    }
  }

  // If we exhausted tool call limit, extract whatever we have
  const lastMsg = messages[messages.length - 1];
  const fallback =
    lastMsg instanceof AIMessage
      ? typeof lastMsg.content === "string"
        ? lastMsg.content
        : ""
      : "";

  return { aiResponse: fallback || "I'm having trouble recalling. Could you rephrase?" };
}
