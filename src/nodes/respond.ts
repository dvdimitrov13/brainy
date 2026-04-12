/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent has one tool: `remember` — searches long-term memory with
 * optional tag filters for precise retrieval.
 *
 * Tags use a two-level system:
 *   - type (fixed): event, decision, preference, fact, goal, plan
 *   - topics (open): free-form keywords like "property", "kitchen", "cycling"
 *
 * The agent can call remember multiple times with different filters
 * to search from different angles (important for counting/listing).
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

const REMEMBER_TOOL = {
  type: "function" as const,
  function: {
    name: "remember",
    description:
      "Search long-term memory for relevant information. Returns conversation summaries from past sessions. " +
      "You can optionally filter by tag type and/or topics to narrow results. " +
      "For counting/listing questions, make multiple calls with different filters to search broadly.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description: "A focused search query. Be specific.",
        },
        type: {
          type: "array" as const,
          items: { type: "string" as const },
          description:
            'Optional type filter: "event", "decision", "preference", "fact", "goal", "plan". ' +
            "E.g., [\"event\"] to find things that happened, [\"decision\"] for choices made.",
        },
        topics: {
          type: "array" as const,
          items: { type: "string" as const },
          description:
            'Optional topic filter: free-form keywords like ["property", "cedar-creek"]. ' +
            "Filters to memories about these topics.",
        },
      },
      required: ["query"],
    },
  },
};

export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const parts: string[] = [];

  if (state.conversationBuffer) {
    parts.push(`Conversation so far:\n${state.conversationBuffer}`);
  }

  const stats = hipporag.getStats();
  if (stats.passages > 0) {
    parts.push(
      `Long-term memory: ${stats.passages} passages, ${stats.entities} entities, ${stats.facts} facts`
    );
  }

  const contextBlock = parts.join("\n\n");

  const systemPrompt = `You are a helpful, friendly assistant with long-term memory.

You have a memory tool: **remember(query, type?, topics?)** — searches past conversations.

Optional filters narrow results:
- **type**: ["event"] for things that happened, ["decision"] for choices, ["preference"] for likes/dislikes, ["fact"] for data, ["goal"] for targets, ["plan"] for intentions
- **topics**: ["property", "kitchen", "cycling"] — keywords for what it's about

How to use:
- For recall questions, call remember with a focused query
- For counting/listing, make multiple calls with different filters to cover all angles
- For complex questions, break them down: first search broadly, then refine
- For casual conversation, just respond directly

Do NOT mention your memory tool. Just respond naturally.

--- Your Memory ---
${contextBlock}
--- End Memory ---`;

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  const maxToolCalls = 5;

  for (let i = 0; i < maxToolCalls; i++) {
    const response = await llm.invoke(messages, {
      tools: [REMEMBER_TOOL],
    });

    const toolCalls = response.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      const aiResponse =
        typeof response.content === "string"
          ? response.content
          : (response.content as Array<{ type: string; text?: string }>)
              .filter((block) => block.type === "text")
              .map((block) => block.text ?? "")
              .join("");

      return { aiResponse };
    }

    messages.push(response);

    for (const toolCall of toolCalls) {
      if (toolCall.name === "remember") {
        const args = toolCall.args as {
          query: string;
          type?: string[];
          topics?: string[];
        };

        const passages = await hipporag.retrieve(
          args.query,
          args.type,
          args.topics
        );

        const result =
          passages.length > 0
            ? passages
                .map((p, idx) => {
                  const tagStr = `[${p.tags.type.join(",")}] [${p.tags.topics.join(",")}]`;
                  return `[Memory ${idx + 1}] ${tagStr}: ${p.text}`;
                })
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

  return {
    aiResponse:
      "I'm having trouble recalling. Could you rephrase?",
  };
}
