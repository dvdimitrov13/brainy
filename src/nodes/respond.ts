/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent has four tools for interacting with its notepad memory:
 *
 *   1. write_notes — add new notes to the notepad (with citations)
 *   2. edit_notes — update an existing section (must read it first)
 *   3. read_notes — read a section's content (or TOC if no section)
 *   4. recall_exchange — fetch raw exchange text from a citation
 *
 * The system prompt always includes the rolling summary + notepad TOC.
 * When mustWriteNotes is set (memory pressure), the agent is forced
 * to call write_notes before responding to the user.
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
import type { NoteOperation } from "../memory/notepad.ts";
import { llm } from "../llm.ts";
import { notepadMemory } from "../singletons.ts";
import { extractJsonFromResponse } from "../utils.ts";

// ══════════════════════════════════════════════
// TOOL DEFINITIONS
// ══════════════════════════════════════════════

const WRITE_NOTES_TOOL = {
  type: "function" as const,
  function: {
    name: "write_notes",
    description:
      "Add new notes to your notepad. Organize by topic using section paths. " +
      "Cite source exchanges with [exchange:ID]. " +
      "Use for recording important facts, decisions, preferences, or any information worth remembering.",
    parameters: {
      type: "object" as const,
      properties: {
        notes: {
          type: "array" as const,
          items: {
            type: "object" as const,
            properties: {
              sectionPath: {
                type: "string" as const,
                description:
                  'Hierarchical path for the note: "Topic/Subtopic". Use "/" for nesting.',
              },
              content: {
                type: "string" as const,
                description:
                  "The note content in markdown. Cite exchanges: [exchange:sess0-turn1]",
              },
              afterSection: {
                type: "string" as const,
                description:
                  "Optional: place this note after the named section.",
              },
            },
            required: ["sectionPath", "content"],
          },
          description: "Array of notes to add to the notepad.",
        },
      },
      required: ["notes"],
    },
  },
};

const EDIT_NOTES_TOOL = {
  type: "function" as const,
  function: {
    name: "edit_notes",
    description:
      "Update an existing section in the notepad. " +
      "You MUST read the section first with read_notes before editing it. " +
      "Use when information needs to be corrected, updated, or expanded.",
    parameters: {
      type: "object" as const,
      properties: {
        section: {
          type: "string" as const,
          description: "The section heading to edit.",
        },
        content: {
          type: "string" as const,
          description: "The new content for this section (replaces existing).",
        },
      },
      required: ["section", "content"],
    },
  },
};

const READ_NOTES_TOOL = {
  type: "function" as const,
  function: {
    name: "read_notes",
    description:
      "Read a section from the notepad. Returns only that section's content, " +
      "not its subsections. If no section specified, returns the table of contents. " +
      "Reading a section enables editing it with edit_notes.",
    parameters: {
      type: "object" as const,
      properties: {
        section: {
          type: "string" as const,
          description:
            "The section heading to read. Omit to get the table of contents.",
        },
      },
      required: [],
    },
  },
};

const RECALL_EXCHANGE_TOOL = {
  type: "function" as const,
  function: {
    name: "recall_exchange",
    description:
      "Fetch the original raw conversation exchange from a notepad citation. " +
      "Use when a note's [exchange:ID] reference doesn't have enough detail.",
    parameters: {
      type: "object" as const,
      properties: {
        id: {
          type: "string" as const,
          description:
            'The exchange ID from a citation, e.g., "sess0-turn3".',
        },
      },
      required: ["id"],
    },
  },
};

const TOOLS = [
  WRITE_NOTES_TOOL,
  EDIT_NOTES_TOOL,
  READ_NOTES_TOOL,
  RECALL_EXCHANGE_TOOL,
];

// ══════════════════════════════════════════════
// RESPOND NODE
// ══════════════════════════════════════════════

/**
 * Generate an AI response with access to notepad tools.
 */
export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  // ── Build system prompt ──
  const parts: string[] = [];

  const rollingSummary = notepadMemory.getRollingSummary();
  if (rollingSummary) {
    parts.push(`Conversation summary: ${rollingSummary}`);
  }

  if (state.conversationBuffer) {
    parts.push(`Recent conversation:\n${state.conversationBuffer}`);
  }

  const toc = notepadMemory.getTOC();
  parts.push(`Your notepad (table of contents):\n${toc}`);

  const contextBlock = parts.join("\n\n");

  // Build forced instruction if memory pressure requires note-writing
  const pressureInstruction = state.mustWriteNotes
    ? `\n\n**IMPORTANT: Your memory buffer is full. You MUST call write_notes NOW to process the pending exchanges into notes before responding to the user. Here are the pending exchanges:\n${state.pendingExchanges.map((e) => `[exchange:${e.id}]\n${e.text}`).join("\n\n")}\n\nProcess these into organized notes, then respond to the user.**`
    : "";

  const systemPrompt = `You are a helpful, friendly assistant with a notepad for long-term memory.

You have four tools:
- **write_notes** — add new notes to your notepad, organized by topic with citations
- **edit_notes** — update an existing section (must read it first)
- **read_notes** — read a section's content, or get the table of contents
- **recall_exchange** — fetch the original conversation from a citation [exchange:ID]

How to use:
- When the user shares important information, write notes to remember it
- For complex recall questions, read relevant notepad sections, then use recall_exchange if you need more detail
- For counting/listing questions, read multiple sections and search thoroughly
- Keep notes dense and factual with citations
- For casual conversation, just respond directly

Do NOT mention your notepad or tools. Just respond naturally.${pressureInstruction}

--- Your Memory ---
${contextBlock}
--- End Memory ---`;

  // ── Tool-calling loop ──
  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  const maxToolCalls = 8; // Higher limit to allow write + read + recall chains
  let notesWritten = false;

  for (let i = 0; i < maxToolCalls; i++) {
    const response = await llm.invoke(messages, { tools: TOOLS });

    const toolCalls = response.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      // Final response
      const aiResponse =
        typeof response.content === "string"
          ? response.content
          : (response.content as Array<{ type: string; text?: string }>)
              .filter((block) => block.type === "text")
              .map((block) => block.text ?? "")
              .join("");

      // If we wrote notes during this cycle, clear pressure
      const updates: Partial<typeof BrainyState.State> = { aiResponse };

      if (notesWritten && state.mustWriteNotes) {
        // Generate rolling summary and compress buffer
        await notepadMemory.updateRollingSummary(state.conversationBuffer);
        notepadMemory.resetReadTracking();

        updates.mustWriteNotes = false;
        updates.pendingExchanges = [];
        updates.conversationBuffer =
          "[Summary]\n" + notepadMemory.getRollingSummary();
      }

      return updates;
    }

    // Execute tool calls
    messages.push(response);

    for (const toolCall of toolCalls) {
      let result = "";

      switch (toolCall.name) {
        case "write_notes": {
          const args = toolCall.args as {
            notes: {
              sectionPath: string;
              content: string;
              afterSection?: string;
            }[];
          };

          const operations: NoteOperation[] = args.notes.map((n) => ({
            sectionPath: n.sectionPath,
            content: n.content,
            afterSection: n.afterSection,
          }));

          notepadMemory.writeNotes(operations);
          notesWritten = true;

          result = `Added ${operations.length} note(s) to the notepad. Updated TOC:\n${notepadMemory.getTOC()}`;
          break;
        }

        case "edit_notes": {
          const args = toolCall.args as {
            section: string;
            content: string;
          };

          const editResult = notepadMemory.editSection(
            args.section,
            args.content
          );

          result = editResult.ok
            ? `Updated section "${args.section}".`
            : editResult.error!;
          break;
        }

        case "read_notes": {
          const args = toolCall.args as { section?: string };

          if (!args.section) {
            result = `Table of Contents:\n${notepadMemory.getTOC()}`;
          } else {
            const content = notepadMemory.readSection(args.section);
            result =
              content !== null
                ? content || "(section exists but has no direct content)"
                : `Section "${args.section}" not found. Available sections:\n${notepadMemory.getTOC()}`;
          }
          break;
        }

        case "recall_exchange": {
          const args = toolCall.args as { id: string };
          const exchange = notepadMemory.getExchange(args.id);
          result =
            exchange ?? `Exchange "${args.id}" not found.`;
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

  // Exhausted tool calls
  return {
    aiResponse:
      "I'm having trouble organizing my thoughts. Could you rephrase?",
  };
}
