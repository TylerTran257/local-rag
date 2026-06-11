# Social Style Retrieval Eval

The social style retrieval eval validates that the RAG pipeline correctly retrieves style memory chunks for social content generation queries. It does not require an LLM -- it measures retrieval quality only.

## What It Measures

The eval ingests a corpus of social style memory chunks (voice rules, hook patterns, CTA patterns, etc.) into an isolated workspace, runs retrieval queries against them, and scores the results on these metrics:

- **Hit rate at K**: fraction of examples where at least one expected chunk was retrieved per requested category.
- **Recall at K**: fraction of all expected chunks (by source label or chunk ID) that appeared in the top-K results.
- **Category coverage**: fraction of requested style categories that returned at least one result.
- **Missing required categories**: count of requested categories that returned zero results.
- **Unexpected categories**: count of categories that returned results but were not requested (or were explicitly excluded).

## How to Run

Using the Python module directly:

```bash
python -m app.evals.social_style_eval
```

Using the convenience script:

```bash
scripts/run_social_style_eval.sh
```

Both forms accept the same flags and forward all arguments.

## Available Flags

| Flag | Description |
|---|---|
| `--eval-set PATH` | Path to a custom eval set JSON file. Defaults to `app/evals/social_style_eval_set.json`. |
| `--example-id ID` | Run only the specified example(s). Can be repeated (e.g. `--example-id voice-example --example-id hook-example`). Defaults to all examples. |
| `--keep-artifacts` | Preserve the isolated eval workspace on disk after the run finishes. Useful for debugging retrieval issues. |
| `--top-k N` | Per-category retrieval limit and scoring cutoff. Defaults to 3. |

## Understanding the Output

A typical run prints a summary like:

```
Social style eval set: /path/to/social_style_eval_set.json
Examples: 4
Retrieval gate: PASSED
Hit@3: 100.00%
Recall@3: 100.00%
Category coverage: 100.00%
Missing required categories: 0
Unexpected categories: 0
```

- **Retrieval gate: PASSED** means every example retrieved its expected chunks with full category coverage and no unexpected categories.
- **Retrieval gate: FAILED** means at least one example missed an expected result. The output lists the failing example IDs under "Retrieval failures".

The process exits with code 0 on PASSED and code 1 on FAILED. Configuration errors (missing eval set files, unknown example IDs) return exit code 2.

## Data Isolation

The eval runner creates an isolated temporary workspace with its own SQLite database and Qdrant vector store. Corpus chunks are ingested into this workspace with a dedicated tenant ID prefix (`eval-tenant:`). No user data is read or modified. The workspace is cleaned up automatically unless `--keep-artifacts` is passed.

## Debugging with --keep-artifacts

When `--keep-artifacts` is passed, the output includes a `Workspace:` line showing the path to the temporary directory. This directory contains:

- `app.db`: the SQLite database with indexed chunk metadata and FTS data.
- `qdrant_data/`: the vector index used during the eval run.

You can inspect these to understand why a particular query did or did not match.
