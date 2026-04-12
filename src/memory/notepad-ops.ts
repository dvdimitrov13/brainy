/**
 * notepad-ops.ts — Pure markdown manipulation functions for the Notepad.
 *
 * These are stateless string operations: parse headings, generate TOC,
 * insert/update/extract sections. No LLM calls, no side effects.
 *
 * The notepad is a standard markdown document where sections are
 * delimited by headings (## for top-level, ### for nested, etc.).
 * The TOC is auto-generated from headings and placed at the top.
 */

/** Rough token estimate (~4 chars per token) */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Parsed section from markdown */
interface ParsedSection {
  heading: string;
  level: number; // 1 = #, 2 = ##, 3 = ###, etc.
  content: string; // content directly under this heading (NOT subsections)
  startIdx: number; // char index where the heading line starts
  endIdx: number; // char index where this section ends (next same/higher level heading or EOF)
  contentEndIdx: number; // char index where the direct content ends (before first subsection)
}

/**
 * Parse markdown into sections.
 *
 * Each section captures:
 * - Its heading text and level
 * - Its direct content (text between this heading and the first subsection or next same-level heading)
 * - Its full range (from heading to just before the next same-or-higher level heading)
 */
export function parseSections(markdown: string): ParsedSection[] {
  const lines = markdown.split("\n");
  const sections: ParsedSection[] = [];
  let currentIdx = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const match = line.match(/^(#{1,6})\s+(.+)$/);

    if (match) {
      const level = match[1]!.length;
      const heading = match[2]!.trim();

      sections.push({
        heading,
        level,
        content: "",
        startIdx: currentIdx,
        endIdx: markdown.length, // will be adjusted
        contentEndIdx: markdown.length, // will be adjusted
      });
    }

    currentIdx += line.length + 1; // +1 for \n
  }

  // Adjust endIdx and contentEndIdx for each section
  for (let i = 0; i < sections.length; i++) {
    const section = sections[i]!;

    // endIdx = start of next same-or-higher level section, or EOF
    for (let j = i + 1; j < sections.length; j++) {
      if (sections[j]!.level <= section.level) {
        section.endIdx = sections[j]!.startIdx;
        break;
      }
    }

    // contentEndIdx = start of first subsection (deeper level), or endIdx
    section.contentEndIdx = section.endIdx;
    for (let j = i + 1; j < sections.length; j++) {
      if (sections[j]!.startIdx >= section.endIdx) break;
      if (sections[j]!.level > section.level) {
        section.contentEndIdx = sections[j]!.startIdx;
        break;
      }
    }

    // Extract direct content (between heading line and contentEndIdx)
    const headingLine =
      "#".repeat(section.level) + " " + section.heading + "\n";
    const contentStart = section.startIdx + headingLine.length;
    section.content = markdown
      .slice(contentStart, section.contentEndIdx)
      .trim();
  }

  return sections;
}

/**
 * Generate a table of contents from markdown headings.
 *
 * Produces indented TOC lines like:
 *   - Company
 *     - Team
 *     - Products
 *   - Personal
 */
export function generateTOC(markdown: string): string {
  const sections = parseSections(markdown);
  if (sections.length === 0) return "";

  const minLevel = Math.min(...sections.map((s) => s.level));

  const lines = sections.map((s) => {
    const indent = "  ".repeat(s.level - minLevel);
    return `${indent}- ${s.heading}`;
  });

  return lines.join("\n");
}

/**
 * Extract a section's direct content by heading name.
 *
 * Returns ONLY the content directly under the heading, NOT subsections.
 * Returns null if the heading is not found.
 */
export function extractSection(
  markdown: string,
  heading: string
): string | null {
  const sections = parseSections(markdown);
  const section = sections.find(
    (s) => s.heading.toLowerCase() === heading.toLowerCase()
  );
  return section ? section.content : null;
}

/**
 * Insert a new section into the markdown.
 *
 * @param markdown — current markdown content
 * @param sectionPath — hierarchical path like "Company/Team"
 * @param content — the note content (without heading)
 * @param afterSection — optional: place after this section heading
 * @returns updated markdown
 */
export function insertSection(
  markdown: string,
  sectionPath: string,
  content: string,
  afterSection?: string
): string {
  const parts = sectionPath.split("/");
  const heading = parts[parts.length - 1]!;
  // Determine heading level from path depth (## for top-level, ### for nested, etc.)
  const level = parts.length + 1; // +1 because # is reserved for doc title
  const headingLine = "#".repeat(level) + " " + heading;
  const sectionBlock = `\n${headingLine}\n\n${content}\n`;

  if (!markdown.trim()) {
    return sectionBlock.trim() + "\n";
  }

  const sections = parseSections(markdown);

  // If afterSection is specified, insert after that section's full range
  if (afterSection) {
    const anchor = sections.find(
      (s) => s.heading.toLowerCase() === afterSection.toLowerCase()
    );
    if (anchor) {
      return (
        markdown.slice(0, anchor.endIdx) +
        sectionBlock +
        markdown.slice(anchor.endIdx)
      );
    }
  }

  // If parent sections exist in the path, insert under the deepest matching parent
  if (parts.length > 1) {
    for (let depth = parts.length - 2; depth >= 0; depth--) {
      const parentHeading = parts[depth]!;
      const parent = sections.find(
        (s) => s.heading.toLowerCase() === parentHeading.toLowerCase()
      );
      if (parent) {
        // Insert at the end of the parent's full range (before next same-level section)
        return (
          markdown.slice(0, parent.endIdx) +
          sectionBlock +
          markdown.slice(parent.endIdx)
        );
      }
    }
  }

  // No anchor found — append to end
  return markdown + sectionBlock;
}

/**
 * Update an existing section's content.
 *
 * Replaces only the direct content of the section, preserving subsections.
 * Returns the original markdown if the heading is not found.
 */
export function updateSection(
  markdown: string,
  heading: string,
  content: string
): string {
  const sections = parseSections(markdown);
  const section = sections.find(
    (s) => s.heading.toLowerCase() === heading.toLowerCase()
  );

  if (!section) return markdown;

  const headingLine =
    "#".repeat(section.level) + " " + section.heading + "\n";
  const contentStart = section.startIdx + headingLine.length;

  return (
    markdown.slice(0, contentStart) +
    "\n" +
    content +
    "\n\n" +
    markdown.slice(section.contentEndIdx)
  );
}

/**
 * Count the number of sections in the markdown.
 */
export function countSections(markdown: string): number {
  return parseSections(markdown).length;
}

/**
 * Build the full notepad markdown with TOC prepended.
 */
export function withTOC(markdown: string): string {
  const toc = generateTOC(markdown);
  if (!toc) return markdown;
  return `# Notepad\n\n## Table of Contents\n\n${toc}\n\n---\n\n${markdown}`;
}
