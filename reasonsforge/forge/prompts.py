"""Prompt templates for forge belief extraction pipeline."""

SUMMARIZE = """\
You are an expert technical writer creating structured study notes.

Given the following documentation page, create a concise summary suitable for \
building domain expertise. Structure your output as:

## <Descriptive Title>
Start with a short, specific title that names the topic (e.g., \
"IAM Role Configuration", "Network Policy Rules", "Cluster Autoscaling"). \
Then one paragraph summarizing what this page covers.

## Key Concepts
Bulleted list of the most important facts, definitions, and concepts.

## Commands and Syntax
Any commands, configuration syntax, or procedures described (with examples).

## Relationships
How this topic connects to other topics in the domain.

## Exam-Relevant Points
Facts that are likely to be tested on a certification exam.

---

SOURCE DOCUMENT:

{content}
"""

SUMMARIZE_CODE = """\
You are an expert technical writer creating structured notes from source code.

Given the following source code file, create a concise summary focused on how \
this code is used in practice. Structure your output as:

## <Descriptive Title>
Start with a short, specific title that names the module or component (e.g., \
"CLI Entry Point", "PDF Chunker", "LLM Invocation Layer"). Then one paragraph \
summarizing what this code does and its role in the project.

## Usage Patterns
How this code is meant to be called or used — entry points, key functions, \
typical invocations. Include code snippets where helpful.

## API and Configuration
Key parameters, options, environment variables, config files, or arguments \
this code accepts.

## Key Behaviors
Important behaviors, error handling, edge cases, or gotchas a user should know about.

## Relationships
How this code connects to other components — what it imports, what calls it, \
what services or systems it interacts with.

---

SOURCE CODE:

{content}
"""

PROPOSE_BELIEFS = """\
You are extracting factual claims from study notes to build a belief registry.

Rules:
- Each belief should be a single, testable factual claim
- Use kebab-case IDs that are descriptive (e.g., rhel9-default-filesystem-xfs)
- Prefer specific facts over vague generalizations
- Include commands, paths, config values when relevant
- Do NOT include opinions or subjective assessments
- Aim for 3-8 beliefs per entry (not every sentence is a belief)
- Set "accept" to true if the claim is well-supported by the source material, \
false if it is vague, speculative, or poorly supported

---

ENTRIES:

{entries}

---

Respond with ONLY this JSON array (no other text):
[{{"id": "<kebab-case-id>", "claim": "<one-line factual claim>", "accept": true, "source": "<path to entry file>", "source_url": "<url from SOURCE_URL in header, or empty string>"}}]
"""
