/**
 * respond.ts — LangGraph node that generates the AI response.
 *
 * The agent has four tools for its flat notepad memory:
 *
 *   1. write_notes — add flat, self-contained note entries with dates + citations
 *   2. edit_notes — update a note by index (must read it first)
 *   3. read_notes — get all notes, or search by keyword
 *   4. recall_exchange — fetch raw exchange from a citation
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
import type { NoteEntry } from "../memory/notepad-ops.ts";
import { llm } from "../llm.ts";
import { notepadMemory } from "../singletons.ts";

// ══════════════════════════════════════════════
// TOOL DEFINITIONS
// ══════════════════════════════════════════════

const WRITE_NOTES_TOOL = {
  type: "function" as const,
  function: {
    name: "write_notes",
    description:
      "Add notes to your notepad. Each note is a flat, self-contained entry with a date, content, and citation. " +
      "Write one note per distinct fact. Each note should include enough context to be findable from any angle. " +
      'Example: "Viewed Cedar Creek property — out of budget, rejected" is findable for "properties viewed", "budget", or "Cedar Creek". ' +
      "Include dates. Only note facts about the user — not generic advice you gave. Keep each note to 1-2 lines.",
    parameters: {
      type: "object" as const,
      properties: {
        notes: {
          type: "array" as const,
          items: {
            type: "object" as const,
            properties: {
              date: {
                type: "string" as const,
                description:
                  "When this fact occurred (from session date or context). E.g., '2023/04/10'",
              },
              content: {
                type: "string" as const,
                description:
                  "Self-contained factual note (1-2 lines). Include enough context to be findable from multiple angles.",
              },
              citations: {
                type: "array" as const,
                items: { type: "string" as const },
                description:
                  'Source exchange IDs. E.g., ["sess0-turn3"]',
              },
            },
            required: ["date", "content", "citations"],
          },
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
      "Update an existing note by its number. You MUST read notes first (via read_notes) before editing. " +
      "Use when information needs correcting or updating.",
    parameters: {
      type: "object" as const,
      properties: {
        note_number: {
          type: "number" as const,
          description: "The note number to edit (from the note list).",
        },
        content: {
          type: "string" as const,
          description: "The updated content for this note.",
        },
      },
      required: ["note_number", "content"],
    },
  },
};

const READ_NOTES_TOOL = {
  type: "function" as const,
  function: {
    name: "read_notes",
    description:
      "Read your notepad. If a search query is provided, returns only matching notes. " +
      "If no query, returns all notes. For counting/listing questions, search broadly or read all notes.",
    parameters: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description:
            "Optional keyword search. Returns notes containing any of these words. Omit to read all notes.",
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
      "Use when a note's [exchange:ID] doesn't have enough detail.",
    parameters: {
      type: "object" as const,
      properties: {
        id: {
          type: "string" as const,
          description: 'The exchange ID, e.g., "sess0-turn3".',
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

export async function respondNode(
  state: typeof BrainyState.State
): Promise<Partial<typeof BrainyState.State>> {
  const parts: string[] = [];

  const rollingSummary = notepadMemory.getRollingSummary();
  if (rollingSummary) {
    parts.push(`Conversation summary: ${rollingSummary}`);
  }

  if (state.conversationBuffer) {
    parts.push(`Recent conversation:\n${state.conversationBuffer}`);
  }

  const noteIndex = notepadMemory.getIndex();
  parts.push(`Your notepad index:\n${noteIndex}`);

  const contextBlock = parts.join("\n\n");

  const pressureInstruction = state.mustWriteNotes
    ? `\n\n**IMPORTANT: Your memory buffer is full. You MUST call write_notes NOW to process the pending exchanges into notes before responding to the user.

Note-taking rules:
- Write one note per distinct fact — self-contained with enough context to be findable from any angle
- Include the date from the session header
- Only note facts about the user: decisions, purchases, goals, events, dates, numbers, preferences
- Do NOT note generic advice you gave — you can regenerate that anytime
- Keep each note to 1-2 lines — cite [exchange:ID] for detail recall later
- Check existing notes first (use read_notes) to avoid duplicating information

Pending exchanges:
${state.pendingExchanges.map((e) => `[exchange:${e.id}]\n${e.text}`).join("\n\n")}

Process these into notes, then respond to the user.**`
    : "";

  const systemPrompt = `You are a helpful, friendly assistant with a notepad for long-term memory.

You have four tools:
- **write_notes** — add new notes (one per fact, self-contained, with date and citation)
- **edit_notes** — update a note by number (must read it first)
- **read_notes** — read all notes or search by keyword
- **recall_exchange** — fetch raw conversation from a citation [exchange:ID]

How to use:
- For recall questions, use read_notes to search your notepad, then recall_exchange for details
- For counting/listing questions, read all notes or search broadly
- Write notes when the user shares important facts (decisions, purchases, dates, goals, preferences)
- Each note should be self-contained — findable from multiple angles

Do NOT mention your notepad or tools. Just respond naturally.${pressureInstruction}

--- Your Memory ---
${contextBlock}
--- End Memory ---`;

  const messages: (SystemMessage | HumanMessage | AIMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(state.userMessage),
  ];

  const maxToolCalls = 8;
  let notesWritten = false;

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

      const updates: Partial<typeof BrainyState.State> = { aiResponse };

      if (notesWritten && state.mustWriteNotes) {
        await notepadMemory.updateRollingSummary(state.conversationBuffer);
        notepadMemory.resetReadTracking();

        updates.mustWriteNotes = false;
        updates.pendingExchanges = [];
        updates.conversationBuffer =
          "[Summary]\n" + notepadMemory.getRollingSummary();
      }

      return updates;
    }

    messages.push(response);

    for (const toolCall of toolCalls) {
      let result = "";

      switch (toolCall.name) {
        case "write_notes": {
          const args = toolCall.args as {
            notes: { date: string; content: string; citations: string[] }[];
          };

          const entries: NoteEntry[] = args.notes.map((n) => ({
            date: n.date,
            content: n.content,
            citations: n.citations,
          }));

          notepadMemory.writeNotes(entries);
          notesWritten = true;

          result = `Added ${entries.length} note(s). Notepad now has ${notepadMemory.getStats().noteCount} notes.`;
          break;
        }

        case "edit_notes": {
          const args = toolCall.args as {
            note_number: number;
            content: string;
          };

          const editResult = notepadMemory.editNote(
            args.note_number,
            args.content
          );
          result = editResult.ok
            ? `Updated note #${args.note_number}.`
            : editResult.error!;
          break;
        }

        case "read_notes": {
          const args = toolCall.args as { query?: string };

          if (args.query) {
            result = notepadMemory.search(args.query);
          } else {
            result = notepadMemory.getAllNotes();
          }
          break;
        }

        case "recall_exchange": {
          const args = toolCall.args as { id: string };
          const exchange = notepadMemory.getExchange(args.id);
          result = exchange ?? `Exchange "${args.id}" not found.`;
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
    aiResponse:
      "I'm having trouble organizing my thoughts. Could you rephrase?",
  };
}
