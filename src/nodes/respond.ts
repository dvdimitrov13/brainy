/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent has two tools:
 *   1. explore_topics(request) — Haiku finds relevant topics from the canonical list
 *   2. remember(query, type?, topics?) — search HippoRAG with optional tag filters
 *
 * The system prompt includes the canonical topic list so the agent knows
 * what topics exist. For complex queries, the agent can use explore_topics
 * to find the right filters before calling remember.
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
import { llm, llmFast } from "../llm.ts";
import { hipporag } from "../singletons.ts";

const EXPLORE_TOPICS_TOOL = {
  type: "function" as const,
  function: {
    name: "explore_topics",
    description:
      "Find which memory topics are relevant to your question. " +
      "Returns a list of matching topic tags you can use to filter your remember calls. " +
      "Use this first for complex or broad questions to discover the right filters.",
    parameters: {
      type: "object" as const,
      properties: {
        request: {
          type: "string" as const,
          description:
            "Describe what you're looking for. E.g., 'all properties the user viewed'",
        },
      },
      required: ["request"],
    },
  },
};

const REMEMBER_TOOL = {
  type: "function" as const,
  function: {
    name: "remember",
    description:
      "Search long-term memory for relevant information. " +
      "Use type and topics filters (from explore_topics or the topic list) to narrow results. " +
      "For counting/listing, make multiple calls with different filters.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description: "A focused search query.",
        },
        type: {
          type: "array" as const,
          items: { type: "string" as const },
          description:
            'Type filter: "event", "decision", "preference", "fact", "goal", "plan".',
        },
        topics: {
          type: "array" as const,
          items: { type: "string" as const },
          description:
            "Topic filter: use exact topic names from the topic list or explore_topics results.",
        },
      },
      required: ["query"],
    },
  },
};

const TOOLS = [EXPLORE_TOPICS_TOOL, REMEMBER_TOOL];

export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const parts: string[] = [];

  if (state.conversationBuffer) {
    parts.push(`Conversation so far:\n${state.conversationBuffer}`);
  }

  const stats = hipporag.getStats();
  const topics = hipporag.getTopics();

  if (stats.passages > 0) {
    parts.push(
      `Long-term memory: ${stats.passages} passages, ${stats.entities} entities`
    );
    if (topics.length > 0) {
      parts.push(`Available topics: ${topics.join(", ")}`);
    }
  }

  const contextBlock = parts.join("\n\n");

  const systemPrompt = `You are a helpful, friendly assistant with long-term memory.

You have two tools:
- **explore_topics(request)** — finds which memory topics match your question. Use first for complex queries.
- **remember(query, type?, topics?)** — searches past conversations. Use type/topics filters for precise results.

Types: event, decision, preference, fact, goal, plan
Topics: use the exact names from the available topics list shown below.

How to use:
- For simple recall: remember with a focused query
- For counting/listing: explore_topics first, then remember with each relevant topic
- For casual conversation: just respond directly

Do NOT mention your tools. Respond naturally.

--- Your Memory ---
${contextBlock}
--- End Memory ---`;

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  const maxToolCalls = 6;

  for (let i = 0; i < maxToolCalls; i++) {
    const response = await llm.invoke(messages, { tools: TOOLS });

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
      let result = "";

      if (toolCall.name === "explore_topics") {
        const args = toolCall.args as { request: string };

        // Ask Haiku which topics from the canonical list are relevant
        const topicList = hipporag.getTopics();
        if (topicList.length === 0) {
          result = "No topics in memory yet.";
        } else {
          const resp = await llmFast.invoke([
            {
              role: "system" as const,
              content: `Given a request and a list of memory topics, return which topics are relevant.

Return ONLY valid JSON: {"relevant_topics": ["topic1", "topic2"]}

Pick only topics that are clearly relevant to the request. Be inclusive rather than exclusive — if in doubt, include it.`,
            },
            {
              role: "user" as const,
              content: `Request: ${args.request}\n\nAvailable topics: ${topicList.join(", ")}`,
            },
          ]);

          const respText =
            typeof resp.content === "string"
              ? resp.content
              : (resp.content as Array<{ type: string; text?: string }>)
                  .filter((b) => b.type === "text")
                  .map((b) => b.text ?? "")
                  .join("");

          try {
            const jsonStr = respText.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
            const parsed = JSON.parse(jsonStr) as {
              relevant_topics: string[];
            };
            result =
              parsed.relevant_topics.length > 0
                ? `Relevant topics: ${parsed.relevant_topics.join(", ")}`
                : "No matching topics found.";
          } catch {
            result = `Available topics: ${topicList.join(", ")}`;
          }
        }
      } else if (toolCall.name === "remember") {
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

        result =
          passages.length > 0
            ? passages
                .map((p, idx) => {
                  const tagStr = `[${p.tags.type.join(",")}] [${p.tags.topics.join(",")}]`;
                  return `[Memory ${idx + 1}] ${tagStr}: ${p.text}`;
                })
                .join("\n\n")
            : "No relevant memories found.";
      }

      messages.push(
        new ToolMessage({
          tool_call_id: toolCall.id ?? `call_${i}`,
          content: result,
        })
      );
    }
  }

  return {
    aiResponse: "I'm having trouble recalling. Could you rephrase?",
  };
}
