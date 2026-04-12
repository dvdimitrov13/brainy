/**
 * test.ts — Multi-turn evaluation of Brainy's dual-memory system.
 *
 * This test simulates a realistic conversation where the user:
 *   1. Feeds the agent several large "documents" of factual information
 *   2. Has some unrelated distractor turns
 *   3. Asks specific recall questions that test different memory capabilities
 *
 * We evaluate:
 *   - DIRECT RECALL: Can the agent recall facts stated explicitly?
 *   - MULTI-HOP RECALL: Can it connect facts across separate passages?
 *     (This is where HippoRAG2's PPR + knowledge graph should shine)
 *   - DISTRACTOR RESILIENCE: Does recall survive after unrelated conversation?
 *   - ENTITY LINKING: Can it find info about an entity mentioned in different contexts?
 *   - SUMMARY COHERENCE: Does the compact memory maintain a sensible narrative?
 *
 * Each recall question has expected keywords that SHOULD appear in the response.
 * The test scores how many keywords the agent successfully recalls.
 */

import { buildGraph } from "./graph.ts";
import { hipporag } from "./singletons.ts";

// ══════════════════════════════════════════════
// TEST DATA — Simulated "documents" the user shares
// ══════════════════════════════════════════════

/**
 * We simulate a user briefing the agent about their company (Aethon Robotics),
 * its team, projects, and technical stack across multiple dense messages.
 * This creates a rich knowledge graph with many interconnected entities.
 */

const DOCUMENT_TURNS: string[] = [
  // Turn 1: Company overview + leadership
  `Let me tell you about my company. I work at Aethon Robotics, headquartered in Austin, Texas.
We were founded in 2021 by Dr. Sarah Chen, who previously led the autonomous systems lab at MIT.
Our CEO is Marcus Webb, a former VP at Boston Dynamics. Our CTO is Priya Ramanathan, who built
the perception stack at Waymo before joining us. We currently have 47 employees and just closed
a Series B round of 38 million dollars led by Sequoia Capital, with participation from
Andreessen Horowitz and our existing investor Khosla Ventures from our Series A.`,

  // Turn 2: Core product + technical architecture
  `Our main product is called Atlas — it's an autonomous mobile robot for warehouse logistics.
Atlas uses a custom SLAM system called NeuroSLAM that combines LiDAR point clouds with
monocular depth estimation from cameras. The perception pipeline runs on NVIDIA Jetson Orin
modules, doing 30fps object detection using a custom YOLOv8 variant we call YOLO-Warehouse
that's been fine-tuned on 2.3 million annotated warehouse images. The navigation stack uses
a hybrid approach: a global planner based on D* Lite for path planning, and a local planner
using Model Predictive Control (MPC) for obstacle avoidance. Communication between robots
uses our mesh networking protocol called HiveLink, operating on the 5GHz band with
sub-10ms latency for fleet coordination.`,

  // Turn 3: Team + current projects
  `Our engineering team is organized into four pods. The Perception pod is led by James
Okafor, who joined from Apple's self-driving car project. The Navigation pod is run by
Lin Zhang, a robotics PhD from CMU. The Fleet Intelligence pod — that's the multi-robot
coordination team — is led by me, and the Hardware pod is managed by Tomás Guerrero,
previously at iRobot. Right now we have three active projects: Project Monarch is
upgrading Atlas to handle mixed palletized and loose-item picking using a new 6-DOF
robotic arm from Universal Robots. Project Firefly is our next-gen LiDAR integration
with Ouster OS1-128 sensors replacing the current Velodyne Puck. And Project Compass
is building a digital twin simulation environment using NVIDIA Isaac Sim for testing
fleet behaviors before deployment.`,

  // Turn 4: Customer deployments + metrics
  `We have four production deployments right now. Our biggest customer is Meridian
Logistics in their Dallas distribution center — they run 24 Atlas units across a
180,000 square foot facility, processing about 12,000 picks per day with a 99.2%
accuracy rate. Our second deployment is with FreshDirect at their Bronx facility —
8 Atlas units for cold-chain grocery fulfillment, operating at temperatures down to
-20°C. We also have a pilot with IKEA at their Älmhult distribution center in Sweden
with 6 units, and a new contract with Mercado Libre for their São Paulo mega-warehouse
starting next quarter with an initial order of 15 units. Our average deployment reduces
labor costs by 34% and increases throughput by 28% compared to manual operations.`,

  // Turn 5: Technical challenges + roadmap
  `The biggest technical challenge we're facing right now is multi-robot coordination
in dynamic environments. When you have 24 robots in a warehouse, deadlock avoidance
becomes a huge problem — we had an incident at Meridian where 7 robots got stuck in a
circular wait at an intersection for 23 minutes before the system detected it. That's
why I'm personally working on a new coordination algorithm based on conflict-based
search (CBS) combined with a priority-based queuing system we call SmartYield. The
algorithm assigns dynamic priorities based on task urgency, battery level, and distance
to destination. Our Q3 roadmap includes: releasing SmartYield v2 with support for up
to 100 robots, launching Project Monarch's arm integration, achieving CE certification
for European markets, and hitting a milestone of 50,000 picks per day across all
deployments. Dr. Chen also wants us to start exploring outdoor last-mile delivery
by Q1 next year — she envisions Atlas operating on sidewalks in downtown areas.`,
];

/** Distractor turns — unrelated conversation to test memory persistence */
const DISTRACTOR_TURNS: string[] = [
  "What's the best way to make sourdough bread? I've been trying but my starter keeps dying.",
  "Can you explain the difference between TCP and UDP? I always mix them up.",
  "I'm thinking about getting a dog. What breeds are good for apartment living?",
];

/**
 * Recall questions with expected keywords that should appear in the response.
 *
 * Each test case has:
 *   - question: what to ask the agent
 *   - expectedKeywords: words/phrases that should appear in a correct answer
 *   - testType: what memory capability this tests
 *   - difficulty: how hard this recall task is
 */
interface RecallTest {
  question: string;
  expectedKeywords: string[];
  testType: "direct" | "multi-hop" | "entity-link" | "detail";
  difficulty: "easy" | "medium" | "hard";
}

const RECALL_TESTS: RecallTest[] = [
  {
    question: "Where is my company headquartered and who founded it?",
    expectedKeywords: ["austin", "texas", "sarah chen", "2021"],
    testType: "direct",
    difficulty: "easy",
  },
  {
    question: "What perception hardware does Atlas use and who leads that team?",
    expectedKeywords: ["jetson", "orin", "james", "okafor", "lidar"],
    testType: "multi-hop",
    difficulty: "medium",
  },
  {
    question:
      "Tell me about the incident at Meridian and what solution I'm building for it.",
    expectedKeywords: ["deadlock", "7 robots", "circular", "smartyield", "cbs"],
    testType: "entity-link",
    difficulty: "medium",
  },
  {
    question:
      "How many Atlas units does our biggest customer run and what metrics do they achieve?",
    expectedKeywords: ["24", "meridian", "12000", "99.2"],
    testType: "detail",
    difficulty: "medium",
  },
  {
    question:
      "What is Project Monarch about and what hardware is it integrating?",
    expectedKeywords: ["arm", "universal robots", "6-dof", "picking"],
    testType: "direct",
    difficulty: "easy",
  },
  {
    question:
      "Who are our investors and how much did we raise in our last round?",
    expectedKeywords: ["sequoia", "38 million", "series b", "andreessen"],
    testType: "direct",
    difficulty: "easy",
  },
  {
    question:
      "What's the connection between Priya Ramanathan and the navigation system on Atlas?",
    expectedKeywords: ["cto", "waymo", "perception"],
    testType: "multi-hop",
    difficulty: "hard",
  },
  {
    question: "Describe all our current customer deployments and where they are located.",
    expectedKeywords: [
      "meridian",
      "dallas",
      "freshdirect",
      "bronx",
      "ikea",
      "mercado libre",
    ],
    testType: "entity-link",
    difficulty: "hard",
  },
  {
    question: "What is our Q3 roadmap? List the milestones.",
    expectedKeywords: ["smartyield", "monarch", "ce certification", "50000"],
    testType: "detail",
    difficulty: "medium",
  },
  {
    question: "What networking protocol do the robots use and what are its specs?",
    expectedKeywords: ["hivelink", "5ghz", "mesh", "10ms"],
    testType: "detail",
    difficulty: "hard",
  },
];

// ══════════════════════════════════════════════
// TEST RUNNER
// ══════════════════════════════════════════════

interface TurnResult {
  turn: number;
  phase: string;
  input: string;
  response: string;
  conversationBuffer: string;
  retrievedContext: string;
  stats: { passages: number; entities: number; facts: number; graphNodes: number };
  recallScore?: { matched: string[]; missed: string[]; score: number };
}

async function runTest() {
  const graph = buildGraph();
  let state: Record<string, unknown> = {};
  const results: TurnResult[] = [];
  let turnNum = 0;

  /**
   * Execute a single turn and collect results.
   */
  async function executeTurn(
    input: string,
    phase: string
  ): Promise<TurnResult> {
    turnNum++;
    const startTime = Date.now();

    const result = await graph.invoke({
      ...state,
      userMessage: input,
    });

    state = {
      conversationBuffer: result.conversationBuffer,
      pendingExchanges: result.pendingExchanges,
      turnCount: result.turnCount,
    };

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const stats = hipporag.getStats();

    const turnResult: TurnResult = {
      turn: turnNum,
      phase,
      input: input.slice(0, 80) + (input.length > 80 ? "..." : ""),
      response: result.aiResponse as string,
      conversationBuffer: result.conversationBuffer as string,
      retrievedContext: (result.retrievedContext as string) || "(none)",
      stats,
    };

    console.log(`  Turn ${turnNum} [${phase}] (${elapsed}s) — KG: ${stats.entities} entities, ${stats.facts} facts, ${stats.passages} passages`);
    return turnResult;
  }

  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║  BRAINY MULTI-TURN EVALUATION                          ║");
  console.log("╚══════════════════════════════════════════════════════════╝\n");

  // ── Phase 1: Feed documents ──
  console.log("━━━ PHASE 1: DOCUMENT INGESTION (5 dense passages) ━━━");
  for (const doc of DOCUMENT_TURNS) {
    const r = await executeTurn(doc, "ingest");
    results.push(r);
  }

  // ── Phase 2: Distractor turns ──
  console.log("\n━━━ PHASE 2: DISTRACTOR TURNS (3 unrelated topics) ━━━");
  for (const distractor of DISTRACTOR_TURNS) {
    const r = await executeTurn(distractor, "distractor");
    results.push(r);
  }

  // ── Phase 3: Recall tests ──
  console.log("\n━━━ PHASE 3: RECALL TESTS (10 questions) ━━━");
  const recallResults: {
    question: string;
    testType: string;
    difficulty: string;
    matched: string[];
    missed: string[];
    score: number;
  }[] = [];

  for (const test of RECALL_TESTS) {
    const r = await executeTurn(test.question, `recall:${test.testType}`);

    // Score the response: check which expected keywords appear
    const responseLower = r.response.toLowerCase();
    const matched = test.expectedKeywords.filter((kw) =>
      responseLower.includes(kw.toLowerCase())
    );
    const missed = test.expectedKeywords.filter(
      (kw) => !responseLower.includes(kw.toLowerCase())
    );
    const score = matched.length / test.expectedKeywords.length;

    r.recallScore = { matched, missed, score };
    results.push(r);
    recallResults.push({
      question: test.question.slice(0, 60) + "...",
      testType: test.testType,
      difficulty: test.difficulty,
      matched,
      missed,
      score,
    });
  }

  // ══════════════════════════════════════════════
  // RESULTS REPORT
  // ══════════════════════════════════════════════

  console.log("\n\n╔══════════════════════════════════════════════════════════╗");
  console.log("║  EVALUATION RESULTS                                      ║");
  console.log("╚══════════════════════════════════════════════════════════╝\n");

  // Overall stats
  const totalScore =
    recallResults.reduce((sum, r) => sum + r.score, 0) / recallResults.length;
  console.log(`Overall Recall Score: ${(totalScore * 100).toFixed(1)}%\n`);

  // Per-question breakdown
  console.log("┌─────────────────────────────────────────────────────────────────────────────┐");
  console.log("│  RECALL TEST RESULTS                                                        │");
  console.log("├─────────────────────────────────────────────────────────────────────────────┤");

  for (const r of recallResults) {
    const emoji = r.score >= 0.75 ? "PASS" : r.score >= 0.5 ? "PARTIAL" : "FAIL";
    console.log(
      `│ [${emoji}] ${r.question}`
    );
    console.log(
      `│        Type: ${r.testType} | Difficulty: ${r.difficulty} | Score: ${(r.score * 100).toFixed(0)}%`
    );
    if (r.matched.length > 0) {
      console.log(`│        Matched: ${r.matched.join(", ")}`);
    }
    if (r.missed.length > 0) {
      console.log(`│        Missed:  ${r.missed.join(", ")}`);
    }
    console.log("│");
  }

  console.log("└─────────────────────────────────────────────────────────────────────────────┘");

  // Score by test type
  console.log("\nScores by test type:");
  for (const type of ["direct", "multi-hop", "entity-link", "detail"] as const) {
    const typeResults = recallResults.filter((r) => r.testType === type);
    if (typeResults.length > 0) {
      const avg =
        typeResults.reduce((sum, r) => sum + r.score, 0) / typeResults.length;
      console.log(`  ${type.padEnd(14)} ${(avg * 100).toFixed(1)}% (${typeResults.length} tests)`);
    }
  }

  // Score by difficulty
  console.log("\nScores by difficulty:");
  for (const diff of ["easy", "medium", "hard"] as const) {
    const diffResults = recallResults.filter((r) => r.difficulty === diff);
    if (diffResults.length > 0) {
      const avg =
        diffResults.reduce((sum, r) => sum + r.score, 0) / diffResults.length;
      console.log(`  ${diff.padEnd(14)} ${(avg * 100).toFixed(1)}% (${diffResults.length} tests)`);
    }
  }

  // Knowledge graph final state
  const finalStats = hipporag.getStats();
  console.log("\nFinal Knowledge Graph:");
  console.log(`  Passages:    ${finalStats.passages}`);
  console.log(`  Entities:    ${finalStats.entities}`);
  console.log(`  Facts:       ${finalStats.facts}`);
  console.log(`  Graph nodes: ${finalStats.graphNodes}`);

  // Conversation buffer final state
  const buf = (state.conversationBuffer as string) || "";
  const bufTokens = Math.ceil(buf.length / 4);
  console.log(`\nConversation Buffer (~${bufTokens} tokens):`);
  console.log(`  "${buf.slice(0, 300)}${buf.length > 300 ? "..." : ""}"`);


  console.log("\n━━━ EVALUATION COMPLETE ━━━\n");
}

runTest().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
