/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent has one memory tool: `remember`. When called, it runs
 * the full HippoRAG pipeline:
 *   1. Recognize — find top-25 triples by cosine similarity, LLM filters
 *   2. Recall — PPR over the knowledge graph using filtered triples as seeds
 *   3. Return passage summaries
 *
 * The agent decides when to use it based on the conversation context.
 * For casual chat it responds directly without calling the tool.
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
      "Search long-term memory for relevant information. " +
      "Finds entity associations and retrieves full conversation summaries " +
      "from past sessions. Use when the user asks about something from " +
      "the past or you need context from prior conversations. " +
      "Pass a focused, specific query.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description:
            "A focused query to search memory for. " +
            "Be specific — e.g., 'user\\'s 5K race time' not 'running'.",
        },
      },
      required: ["query"],
    },
  },
};

/**
 * Generate an AI response with access to the remember tool.
 */
export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const memoryParts: string[] = [];

  if (state.conversationBuffer) {
    memoryParts.push(
      `Conversation so far:\n${state.conversationBuffer}`
    );
  }

  const memoryBlock = memoryParts.join("\n\n");

  const systemPrompt = `You are a helpful, friendly assistant with long-term memory.

You have a memory tool: **remember** — searches past conversations for relevant information. Use it when the user asks about something from the past or you need context from prior conversations. For casual conversation, just respond directly.

Do NOT mention your memory tool or system. Just respond naturally.

${memoryBlock ? `--- Current Session ---\n${memoryBlock}\n--- End Session ---` : "(New conversation — no prior context in this session.)"}`;

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  const maxToolCalls = 3;

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
        const query = (toolCall.args as { query: string }).query;

        // Full pipeline: recognize (top-25 + LLM filter) → recall (PPR)
        const passages = await hipporag.retrieve(query);

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
