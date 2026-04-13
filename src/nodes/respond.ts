/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent sees auto-recognized triples in the prompt (from retrieve node)
 * and has 4 tools:
 *
 *   1. recognize(query, type?, topics?) — find triples with a custom query
 *   2. recall(query) — PPR using last recognized triples → passages
 *   3. remember(query, type?, topics?) — convenience: recognize + recall
 *   4. explore_topics(request) — find relevant topic filters via Haiku
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
import { llm, llmFast } from "../llm.ts";
import { hipporag } from "../singletons.ts";

// ══════════════════════════════════════════════
// TOOL DEFINITIONS
// ══════════════════════════════════════════════

const RECOGNIZE_TOOL = {
  type: "function" as const,
  function: {
    name: "recognize",
    description:
      "Search for entity associations in long-term memory. Returns triples (subject, predicate, object). " +
      "Use when the auto-recognized associations aren't enough or you want to search from a different angle. " +
      "After recognize, call recall to get full passages seeded by these triples.",
    parameters: {
      type: "object" as const,
      properties: {
        query: { type: "string" as const, description: "Focused search query." },
        type: {
          type: "array" as const,
          items: { type: "string" as const },
          description: 'Optional type filter: "event","decision","preference","fact","goal","plan".',
        },
        topics: {
          type: "array" as const,
          items: { type: "string" as const },
          description: "Optional topic filter.",
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
      "Retrieve full conversation summaries from long-term memory using the most recently recognized triples as seeds. " +
      "Call this after seeing relevant triples (auto-recognized or from a recognize call) to get the actual details.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description: "The search query for passage ranking.",
        },
      },
      required: ["query"],
    },
  },
};

const REMEMBER_TOOL = {
  type: "function" as const,
  function: {
    name: "remember",
    description:
      "Full memory search from scratch: finds triples AND retrieves passages in one call. " +
      "Use when you want results without calling recognize + recall separately. " +
      "For counting/listing, make multiple calls with different filters.",
    parameters: {
      type: "object" as const,
      properties: {
        query: { type: "string" as const, description: "Search query." },
        type: {
          type: "array" as const,
          items: { type: "string" as const },
          description: 'Type filter: "event","decision","preference","fact","goal","plan".',
        },
        topics: {
          type: "array" as const,
          items: { type: "string" as const },
          description: "Topic filter.",
        },
      },
      required: ["query"],
    },
  },
};

const EXPLORE_TOPICS_TOOL = {
  type: "function" as const,
  function: {
    name: "explore_topics",
    description:
      "Discover which memory topics exist and are relevant to your question. " +
      "Use before recognize/remember to find the right topic filters.",
    parameters: {
      type: "object" as const,
      properties: {
        request: {
          type: "string" as const,
          description: "What you're looking for.",
        },
      },
      required: ["request"],
    },
  },
};

const TOOLS = [RECOGNIZE_TOOL, RECALL_TOOL, REMEMBER_TOOL, EXPLORE_TOPICS_TOOL];

// ══════════════════════════════════════════════
// RESPOND NODE
// ══════════════════════════════════════════════

export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const parts: string[] = [];

  if (state.conversationBuffer) {
    parts.push(`Conversation so far:\n${state.conversationBuffer}`);
  }

  if (state.recognizedTriples) {
    parts.push(
      `Memory associations (auto-recognized):\n${state.recognizedTriples}`
    );
  }

  const stats = hipporag.getStats();
  if (stats.passages > 0) {
    parts.push(
      `Long-term memory: ${stats.passages} passages, ${stats.entities} entities`
    );
  }

  const contextBlock = parts.join("\n\n");

  const systemPrompt = `You are a helpful, friendly assistant with long-term memory.

You have four memory tools:
- **recognize(query)** — find entity associations (triples). Use to search from a different angle.
- **recall(query)** — retrieve full passages using the last recognized triples as seeds. Call after seeing relevant triples.
- **remember(query)** — full search from scratch (recognize + recall in one call).
- **explore_topics(request)** — discover available topic tags for filtering.

Memory associations are auto-recognized each turn and shown above. If they're relevant, call **recall** to get the full details. If you need to search differently, use **recognize** or **remember**.

For counting/listing questions: use explore_topics first, then remember with different topic filters.
For casual conversation: just respond directly.

Do NOT mention your memory tools. Respond naturally.

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

      switch (toolCall.name) {
        case "recognize": {
          const args = toolCall.args as {
            query: string;
            type?: string[];
            topics?: string[];
          };
          const triples = await hipporag.recognize(
            args.query,
            args.type,
            args.topics
          );
          result =
            triples.length > 0
              ? triples
                  .map(
                    (t, idx) =>
                      `${idx + 1}. (${t.subject}, ${t.predicate}, ${t.object})`
                  )
                  .join("\n")
              : "No associations found.";
          break;
        }

        case "recall": {
          const args = toolCall.args as { query: string };
          // Uses lastRecognizedTriples from the most recent recognize call
          const passages = await hipporag.recall(args.query);
          result = formatPassages(passages);
          break;
        }

        case "remember": {
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
          result = formatPassages(passages);
          break;
        }

        case "explore_topics": {
          const args = toolCall.args as { request: string };
          const topicList = hipporag.getTopics();

          if (topicList.length === 0) {
            result = "No topics in memory yet.";
          } else {
            const resp = await llmFast.invoke([
              {
                role: "system" as const,
                content: `Given a request and topic list, return relevant topics as JSON: {"relevant_topics": ["t1","t2"]}. Be inclusive.`,
              },
              {
                role: "user" as const,
                content: `Request: ${args.request}\nTopics: ${topicList.join(", ")}`,
              },
            ]);

            const respText =
              typeof resp.content === "string" ? resp.content : "";
            try {
              const jsonStr = respText
                .replace(/```json\n?/g, "")
                .replace(/```\n?/g, "")
                .trim();
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
          break;
        }

        default:
          result = `Unknown tool: ${toolCall.name}`;
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

/** Format passages with tags for the LLM */
function formatPassages(
  passages: import("../hipporag/types.ts").Passage[]
): string {
  if (passages.length === 0) return "No relevant memories found.";
  return passages
    .map((p, idx) => {
      const tagStr = `[${p.tags.type.join(",")}] [${p.tags.topics.join(",")}]`;
      return `[Memory ${idx + 1}] ${tagStr}: ${p.text}`;
    })
    .join("\n\n");
}
