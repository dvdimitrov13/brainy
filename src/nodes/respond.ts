/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * This node assembles the prompt from memory context and calls the LLM.
 * The prompt includes:
 *   - System prompt with injected memories
 *   - The conversation buffer (real turns or compressed summary)
 *   - The current user message
 *
 * The conversation buffer gives the LLM real message history when
 * turns are recent (under memory pressure threshold), and compressed
 * summaries for older history. HippoRAG passages supplement this
 * with relevant long-term memories retrieved via graph search.
 *
 * Graph position: START → retrieve → [respond] → memorize → END
 */

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { BrainyState } from "../state.ts";
import { llm } from "../llm.ts";

/**
 * Generate an AI response using memory-augmented context.
 *
 * @param state — current graph state with userMessage + memory context
 * @returns partial state update with aiResponse populated
 */
export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  // ── Build the memory block for the system prompt ──
  // Only include sections that have content (avoid empty blocks)
  const memoryParts: string[] = [];

  if (state.conversationBuffer) {
    memoryParts.push(
      `Conversation so far:\n${state.conversationBuffer}`
    );
  }

  if (state.retrievedContext) {
    memoryParts.push(
      `Relevant long-term memories:\n${state.retrievedContext}`
    );
  }

  const memoryBlock = memoryParts.join("\n\n");

  // ── Assemble the system prompt ──
  const systemPrompt = `You are a helpful, friendly assistant with a long-term memory system.
You can recall details from earlier in the conversation through your memory stores.
Use the memories below to maintain continuity and recall relevant context.
Do NOT explicitly mention your memory system or say things like "according to my memories".
Just respond naturally as if you remember the conversation yourself.

${memoryBlock ? `--- Your Memories ---\n${memoryBlock}\n--- End Memories ---` : "(No memories stored yet — this is the beginning of the conversation.)"}`;

  // ── Call the LLM with system prompt + current user message ──
  const response = await llm.invoke([
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ]);

  // Extract the response text
  const aiResponse =
    typeof response.content === "string"
      ? response.content
      : (response.content as Array<{ type: string; text?: string }>)
          .filter((block) => block.type === "text")
          .map((block) => block.text ?? "")
          .join("");

  return { aiResponse };
}
