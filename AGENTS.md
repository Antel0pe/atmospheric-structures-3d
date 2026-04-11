<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Python Environment

Use the conda environment named `atmospheric-structures-3d` for Python work in this repo.

# Notes Memory

The repo contains a local-only `notes/` directory that should be treated as working memory for ongoing collaboration, not as polished project documentation.

Purpose:
- capture the user's evolving goals, preferences, and reasoning
- preserve rejected ideas so they are not revisited without context
- track how conclusions changed and what observations or data caused the change
- maintain short meteorological and visualization notes that help future work
- preserve durable technical lessons from difficult bugs so future agents can reuse them

How to use it:
- start with `notes/instructions.md`
- use `notes/index.md` as the directory map
- read `notes/current-direction.md` before making product or design decisions
- read `notes/evolution.md` to understand how the user's thinking has changed
- consult `notes/open-questions.md` when brainstorming or deciding what to investigate next
- consult `notes/technical-notes.md` before deep debugging or when an issue resembles a past rendering/data failure
- update the notes when the user's preferences, reasoning, or priorities change in a meaningful way

Conventions:
- prefer appending updates over deleting history
- when an idea is superseded, mark it as changed and explain why
- separate current best direction from open experiments and discarded ideas
- keep entries concise and practical
- if a view changed because of a data observation or a viewer experiment, record that cause
- if a non-obvious bug is solved, record the symptom, diagnosis path, root cause, and fix in `notes/technical-notes.md`
- keep all references repo-relative; never write full local filesystem paths into the notes

When to update:
- after the user clearly changes their mind
- after discovering a meaningful constraint in the data or rendering
- after identifying a new success criterion or product framing
- after ruling out an approach for scientific or product reasons
- after solving a difficult technical issue that could plausibly recur

Treat the notes as an active tool. Before major brainstorming, planning, or implementation work in this repo, check whether the notes contain relevant context and refresh them if needed.

# Privacy

Never include the user's full local filesystem path in any public or shareable artifact. This includes committed files, documentation, commit messages, pull requests, issues, comments, generated reports, screenshots, and any other text that may leave the local machine. Use repo-relative paths, `~`, or sanitized placeholders instead.
