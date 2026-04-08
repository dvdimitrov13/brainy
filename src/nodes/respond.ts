/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * This node assembles the prompt from memory context and calls the LLM.
 * Crucially, it uses ONLY:
 *   - System prompt with injected memories
 *   - The current user message (singular — no history!)
 *
 * This is the "always turn 1" design: the LLM sees no prior messages.
 * All continuity comes from the compact summary and retrieved passages
 * that are injected into the system prompt.
 *
 * The prompt stays under ~3.5K tokens regardless of conversation length
 * because the compact summary is always one sentence and we retrieve
 * at most 3 passages.
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

  if (state.compactSummary) {
    memoryParts.push(`Conversation summary: ${state.compactSummary}`);
  }

  if (state.metaSummary) {
    memoryParts.push(`Overall narrative: ${state.metaSummary}`);
  }

  if (state.retrievedContext) {
    memoryParts.push(`Relevant memories:\n${state.retrievedContext}`);
  }

  const memoryBlock = memoryParts.join("\n\n");

  // ── Assemble the system prompt ──
  const systemPrompt = `You are a helpful, friendly assistant with a long-term memory system.
You can recall details from earlier in the conversation through your memory stores.
Use the memories below to maintain continuity and recall relevant context.
Do NOT explicitly mention your memory system or say things like "according to my memories".
Just respond naturally as if you remember the conversation yourself.

${memoryBlock ? `--- Your Memories ---\n${memoryBlock}\n--- End Memories ---` : "(No memories stored yet — this is the beginning of the conversation.)"}`;

  // ── Call the LLM with ONLY system prompt + current user message ──
  // No message history! This is the "always turn 1" design.
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
