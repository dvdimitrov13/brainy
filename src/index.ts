/**
 * index.ts — Entry point: interactive terminal chat loop.
 *
 * This is where the user interacts with Brainy. It:
 *   1. Builds the LangGraph pipeline
 *   2. Reads user input from stdin
 *   3. Invokes the graph with each message
 *   4. Prints the AI response
 *   5. Loops until the user types 'quit'
 *
 * State management:
 *   We maintain a `currentState` object that carries the persistent
 *   fields (compactSummary, metaSummary, turnCount) across turns.
 *   Each turn, we pass this state PLUS the new userMessage to the graph.
 *   The graph returns updated state with the new aiResponse + updated memories.
 *
 *   Note: there's NO message accumulation. Each turn overwrites
 *   userMessage and aiResponse. The only things that persist are
 *   the summary fields and turn count.
 *
 * TS note for Python devs:
 *   - `readline` module provides line-by-line input from stdin.
 *     Python equivalent: `input()` in a while loop.
 *   - `createInterface` sets up the input/output streams.
 *   - `rl.question(prompt, callback)` is async — it calls the callback
 *     when the user presses Enter. This is event-driven, unlike Python's
 *     blocking `input()`.
 *   - `process.stdout.write()` is like `print(..., end='')` in Python.
 */

import * as readline from "node:readline";
import { buildGraph } from "./graph.ts";
import { hipporag } from "./singletons.ts";

async function main() {
  // Build the LangGraph pipeline
  const graph = buildGraph();

  // Persistent state that carries across turns
  // (conversation buffer + turn count — buffer holds real turns until
  // memory pressure triggers summarization)
  let currentState: Record<string, unknown> = {};

  // Set up readline for terminal input
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║  Brainy — Hippocampus-Inspired Memory Agent     ║");
  console.log("║  Dual memory: Compact Summary + HippoRAG2       ║");
  console.log("║  Type 'quit' to exit, 'stats' for memory stats  ║");
  console.log("╚══════════════════════════════════════════════════╝");
  console.log();

  /**
   * Recursive function that prompts the user and processes their input.
   *
   * We use a recursive callback pattern because readline.question()
   * is callback-based. Each call handles one turn, then calls itself
   * for the next turn.
   *
   * Python equivalent:
   *   while True:
   *       user_input = input("You: ")
   *       if user_input == "quit": break
   *       response = graph.invoke(...)
   *       print(f"Brainy: {response}")
   */
  const askQuestion = (): void => {
    rl.question("You: ", async (input: string) => {
      const trimmed = input.trim();

      // Handle exit
      if (trimmed.toLowerCase() === "quit") {
        console.log("\nGoodbye! Your memories will be lost (in-memory only).");
        rl.close();
        return;
      }

      // Handle empty input
      if (!trimmed) {
        askQuestion();
        return;
      }

      // Handle stats command (for debugging)
      if (trimmed.toLowerCase() === "stats") {
        const stats = hipporag.getStats();
        console.log("\n--- Memory Stats ---");
        console.log(`Passages:  ${stats.passages}`);
        console.log(`Entities:  ${stats.entities}`);
        console.log(`Facts:     ${stats.facts}`);
        console.log(`Turn count: ${(currentState.turnCount as number) ?? 0}`);
        const buf = (currentState.conversationBuffer as string) || "";
        const bufTokens = Math.ceil(buf.length / 4);
        console.log(`Buffer:    ~${bufTokens} / 1024 tokens`);
        console.log("---\n");
        askQuestion();
        return;
      }

      try {
        // Invoke the graph with current state + new user message
        // LangGraph merges the input with existing state using reducers
        const result = await graph.invoke({
          ...currentState,
          userMessage: trimmed,
        });

        // Update our persistent state for the next turn
        // We carry forward: conversationBuffer, pendingExchanges, turnCount
        // We do NOT carry: userMessage, aiResponse, retrievedContext
        // (those are per-turn and get overwritten)
        currentState = {
          conversationBuffer: result.conversationBuffer,
          pendingExchanges: result.pendingExchanges,
          turnCount: result.turnCount,
        };

        // Print the AI response
        console.log(`\nBrainy: ${result.aiResponse}\n`);
      } catch (error) {
        console.error("\n[Error]", error);
        console.log("(Something went wrong — try again)\n");
      }

      // Loop: ask for the next message
      askQuestion();
    });
  };

  // Start the conversation loop
  askQuestion();
}

// Run the main function
main().catch(console.error);
