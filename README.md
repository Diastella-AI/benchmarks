# Diastella benchmarks

[Diastella](https://diastella.com) keeps a failure memory for software projects: every past bug report linked to the
merged change that fixed it, the files that change touched and who wrote it. This repository holds the two
measurements behind the numbers on [diastella.com/agents](https://diastella.com/agents), with the code that produced
them and the result files, so that anyone can check them or run them again.

Both are replays on public GitHub history. No language model and no agent is involved; the only model is the small
embedding model behind search by meaning (all-MiniLM-L6-v2). Every search runs in [Qdrant](https://qdrant.tech).

## What was measured

### 1. Does the memory point at the file a real fix changed?

For each past bug that was closed by a merged change, the memory is cut back to the day the bug was reported. The
bug's own words are the question. A search scores when a fix it returns changed a source file that the real fix
changed. 9,863 bug reports were read from 24 projects; 3,183 bugs in 22 projects passed the filters and were replayed.

Right file among the first five returned:

| Method | Asked with the report | Asked with the title alone |
|---|---|---|
| Diastella's search (meaning and words, fused) | 34.4% | 37.3% |
| The same search, read as the hosted tool reads it | 35.2% | 36.8% |
| Meaning alone | 35.6% | 35.8% |
| Keywords alone | 27.8% | 34.6% |
| Rule: the files fixed most often | 16.4% | 16.4% |
| Rule: the most recent fixes | 12.4% | 12.4% |
| Random earlier fixes | 8.4% | 8.4% |
| Best possible from history | 64.5% | 64.5% |

What this shows: the search finds a right file about twice as often as the best of the three simple rules, and that
lead holds when whole projects, not single bugs, are the unit. What it does not show: the fused search is **not**
clearly better than meaning alone. It is 1.2 points below with reports and 1.5 above with titles, and neither
difference holds with projects as the unit.

A change we tested and did not make: weighting the search toward meaning (`tune_search.py`). A setting was chosen on
11 projects and checked on the other 11. Two settings tied when chosen, and on the held-out half neither could be told
from the search as it is, so the result is recorded as inconclusive and the search was left unchanged.

### 2. Is the answer in another repository?

A report filed in one project sometimes turns out to be a bug in a project it uses. GitHub records the link when the
downstream report mentions the upstream bug, and those links are the ground truth here: nobody labelled anything for
this test. Of 9,863 upstream bug reports, 1,021 are mentioned by an issue in another repository. The fair test is the
297 cases where the link was made on a later day than the report, so its writer had not yet named the upstream bug.
The question is the downstream report's title alone.

Linked upstream bug among the first five results, on those 297:

| Search | First five |
|---|---|
| Diastella's search, over the project's recent bug-labelled reports | 41.8% |
| Keywords alone, same reports | 37.7% |
| Meaning alone, same reports | 36.0% |
| Rule: the most recently reported bugs | 18.5% |
| By chance | 4.6% |
| GitHub's hybrid issue search, limited to the same bug label | 21.6% |
| GitHub's hybrid issue search, across all of the project's issues | 14.8% |

**What must be said with those numbers**

- The pools differ. Diastella holds at most the 480 most recent bug-labelled reports of a project; GitHub searches
  every issue opened before the date. On the 65 questions where both searched exactly the same reports: 49% against 31%.
- Most of the lead comes from the small prepared pool, not from a cleverer ranking: keywords alone score 37.7% on
  the same questions.
- A link is not proof of cause. Of 15 pairs read one by one, 12 were a real downstream symptom with an upstream cause.
- GitHub's search was given the title word for word, once. A person would rephrase a question that returned little.
- These are only cases someone eventually linked. The test says nothing about how often a failure has a known upstream
  cause, or about false alarms when it has none.

Neither measurement shows that a person or a coding agent fixes bugs faster with this memory. That has not been
measured, and Diastella does not claim it.

Every limit is listed in full inside each result file, under `limits`.

## Run it yourself

You need Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, and the GitHub command `gh`, signed in (only public
data is read).

```
docker compose up -d                          # a local Qdrant on port 6333
uv sync

uv run python benchmark_memory.py fetch       # read the 24 projects from GitHub, kept under .cache/
uv run python benchmark_memory.py run         # store, replay, write results/memory-<date>.json

uv run python benchmark_upstream.py links     # read the cross-repository links from GitHub
uv run python benchmark_upstream.py run       # replay, write results/upstream-<date>.json
uv run python benchmark_upstream.py github 297   # the same questions to GitHub's own search (slow: its rate limits)

uv run python tune_search.py collect          # the weighting test
uv run python tune_search.py choose
```

Reading from GitHub is the slow part and is bounded by GitHub's rate limits. Your figures will differ a little from
the ones above: projects have gained and closed bug reports since 17 November 2026, and the search service does not
fix the order of records whose scores are exactly equal (about three bugs in three thousand change between two runs
on the same data).

## What is in this repository

| Path | What it is |
|---|---|
| `benchmark_memory.py` | The first measurement |
| `benchmark_upstream.py` | The second measurement, and the comparison with GitHub's search |
| `tune_search.py` | The weighting test, with its rules written before anything was scored |
| `diastella_bench/reader.py` | Reading a project's bug reports and fixes from GitHub |
| `diastella_bench/search.py` | The search in Qdrant: one collection, a tenant key per project, dense and BM25 vectors fused by reciprocal rank fusion |
| `diastella_bench/clean.py` | How the hosted tool cleans a piece of text |
| `results/` | The result files of 17 November 2026, as published |

`reader.py` and `search.py` are the parts of the hosted service that these measurements call, cut out unchanged in
what they do. The hosted service itself (the map of a repository, the AI agents, the address for coding agents) is not
in this repository.

## How this was checked, and what has not been done

- Both measurements were run again on 18 November 2026 from this repository alone, with none of the hosted service's
  code, on the GitHub data read the day before. Every figure above came out the same, except three in the first
  table that moved by 0.1 to 0.2 points (the order among equal scores, on one to three bugs of 3,183).
- Each result was checked by separate automated reviews: AI reviewers that did not write the code recomputed the
  figures and read the method for flaws. Several findings changed the published wording and are recorded in the
  result files. Where a file says "review" or "reviewers", that is what is meant.
- **No outside person has reviewed this work yet.** If you find a flaw, an issue here is the most useful thing you
  can give us.
- Two things found on the way that users of Qdrant may want to know. Weighted reciprocal rank fusion scored a record
  1 / (k − 1 + rank / weight) in the version used (1.19.1); `tune_search.py verify` checks this against the service.
  And the keyword statistics behind BM25 are counted over the whole collection, not per tenant, so results shift
  slightly with the set of projects stored.

## How it was made

Diastella is built by one founder with the help of an AI coding assistant (Claude, through Claude Code), which wrote
much of this code under his direction. The measurements were designed to be checkable for that reason.

## Licence

Apache-2.0. See `LICENSE`.

Questions and corrections: open an issue, or write to hello@diastella.com.
