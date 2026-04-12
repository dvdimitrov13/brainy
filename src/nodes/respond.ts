/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent has two memory tools:
 *
 *   1. recognize — surface entity associations from long-term memory.
 *      The agent formulates a contextualized query, gets back triples
 *      (lightweight connections between entities).
 *
 *   2. recall — retrieve full passage summaries from long-term memory.
 *      Uses the triples from recognize as seeds for PPR over the
 *      knowledge graph. Returns dense exchange summaries.
 *
 * No memory is injected automatically — the agent decides when to
 * search its memory based on the conversation context.
 *
 * Graph position: START → [respond] → memorize → END
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
import type { Triple } from "../hipporag/types.ts";

const RECOGNIZE_TOOL = {
  type: "function" as const,
  function: {
    name: "recognize",
    description:
      "Search long-term memory for relevant entity associations. " +
      "Returns relationship triples (subject, predicate, object) that " +
      "connect to your query. Use this first to check what you might " +
      "remember about a topic. Pass a focused, contextualized query.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description:
            "A focused query to search memory associations. " +
            "Be specific — e.g., 'user\\'s 5K race time' not 'running'.",
        },
      },
      required: ["query"],
    },
  },
};

const RECALL_TOOL = {
  type: "function" as const,
  function: {
    name: "recall",
    description:
      "Retrieve full conversation summaries from long-term memory. " +
      "Use AFTER recognize — this does a deep search using the " +
      "associations found. Returns the actual conversation content " +
      "as dense summaries. Pass the same or refined query.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description:
            "The query to search long-term memory for. " +
            "Usually the same query used for recognize, or a refined version.",
        },
      },
      required: ["query"],
    },
  },
};

const TOOLS = [RECOGNIZE_TOOL, RECALL_TOOL];

/**
 * Generate an AI response with access to memory tools.
 *
 * The agent decides whether to use recognize/recall based on
 * the conversation context. Tool calls are handled in a loop
 * until the agent produces a final text response.
 */
export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  // ── Build system prompt ──
  const memoryParts: string[] = [];

  if (state.conversationBuffer) {
    memoryParts.push(
      `Conversation so far:\n${state.conversationBuffer}`
    );
  }

  const memoryBlock = memoryParts.join("\n\n");

  const systemPrompt = `You are a helpful, friendly assistant with long-term memory.

You have two memory tools:
1. **recognize** — search for entity associations in memory. Returns relationship triples. Use this to check if you remember something relevant.
2. **recall** — retrieve full conversation summaries. Use after recognize to get the actual details. Uses the associations found to do a deep search.

Workflow: if the user asks about something from the past, first recognize to find associations, then recall to get the details. For casual conversation or topics you don't need memory for, just respond directly.

Do NOT mention your memory tools or system. Just respond naturally.

${memoryBlock ? `--- Current Session ---\n${memoryBlock}\n--- End Session ---` : "(New conversation — no prior context in this session.)"}`;

  // ── Tool-calling loop ──
  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  // Track triples from recognize for use in recall
  let lastRecognizedTriples: Triple[] = [];

  const maxToolCalls = 5; // Safety limit

  for (let i = 0; i < maxToolCalls; i++) {
    const response = await llm.invoke(messages, { tools: TOOLS });

    const toolCalls = response.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      // Final response — no more tool calls
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
      const query = (toolCall.args as { query: string }).query;

      if (toolCall.name === "recognize") {
        const triples = await hipporag.recognize(query);
        lastRecognizedTriples = triples;

        const result =
          triples.length > 0
            ? triples
                .map(
                  (t, idx) =>
                    `${idx + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
                )
                .join("\n")
            : "No relevant associations found in memory.";

        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? `call_${i}`,
            content: result,
          })
        );
      } else if (toolCall.name === "recall") {
        // Use triples from the last recognize call as PPR seeds
        const passages = await hipporag.recall(
          query,
          lastRecognizedTriples
        );

        const result =
          passages.length > 0
            ? passages
                .map((p, idx) => `[Memory ${idx + 1}]: ${p.text}`)
                .join("\n\n")
            : "No relevant memories found.";

        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? `call_${i}`,
            content: result,
          })
        );
      }
    }
  }

  // Exhausted tool call limit — extract whatever we have
  const lastMsg = messages[messages.length - 1];
  const fallback =
    lastMsg instanceof AIMessage
      ? typeof lastMsg.content === "string"
        ? lastMsg.content
        : ""
      : "";

  return {
    aiResponse:
      fallback || "I'm having trouble recalling. Could you rephrase?",
  };
}
