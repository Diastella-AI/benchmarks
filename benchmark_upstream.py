"""Is the answer in another repository, and can a project's failure memory find it from there?

A report filed in one project often turns out to be a bug of something that project depends on. GitHub records the
link when the downstream report (or a comment on it) mentions the upstream bug. Those links are the ground truth here:
nobody labelled anything for this test.

For every such pair where the upstream bug was already reported when the downstream report was written, the upstream
project's memory is cut back to the day of the downstream report, the downstream reporter's own words are the
question (their title; links and issue numbers removed), and each way of searching returns upstream bug reports. A
search "hits" when the linked upstream bug is among the first reports returned.

    uv run python benchmark_upstream.py links     # read the links from GitHub (cached under .cache/)
    uv run python benchmark_upstream.py run       # replay, write results/upstream-<date>.json
    uv run python benchmark_upstream.py github 297  # put the same questions to GitHub's own issue search (slow: its limits)

Needs the collection that benchmark_memory.py builds (run that first), a local Qdrant and the `gh` command, signed in.
No language model."""
import datetime
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import benchmark_memory as bm
from diastella_bench import search as recall

LINKS = bm.CACHE.parent / "upstream"
PER_QUERY, EVENTS = 12, 30
ONE = """i%d: issue(number: %d) { number timelineItems(itemTypes: [CROSS_REFERENCED_EVENT], first: %d) { nodes { ... on CrossReferencedEvent {
  isCrossRepository referencedAt source { __typename ... on Issue { number title bodyText createdAt state url
    author { login __typename } repository { nameWithOwner isPrivate isFork } } } } } } }"""


def _ask(repo: str, numbers: list[int]) -> dict:
    owner, name = repo.split("/")
    query = "query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) { %s } }" % " ".join(ONE % (n, n, EVENTS) for n in numbers)
    for attempt in range(4):
        out = subprocess.run(["gh", "api", "graphql", "-f", f"query={query}", "-F", f"owner={owner}", "-F", f"name={name}"], capture_output=True, text=True)
        try:
            got = json.loads(out.stdout or "{}").get("data", {}).get("repository")
        except json.JSONDecodeError:
            got = None
        if got:
            return got
        time.sleep(4 * (attempt + 1))
    return {}


def links(repo: str) -> dict | None:
    """For each bug report of a project: the issues in OTHER public repositories that mention it."""
    path = LINKS / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    data = bm.fetch(repo)
    if not data:
        return None
    numbers = [i["number"] for i in data["incidents"]]
    batches = [numbers[k:k + PER_QUERY] for k in range(0, len(numbers), PER_QUERY)]
    found: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for got in pool.map(lambda b: _ask(repo, b), batches):
            for node in got.values():
                if not node:
                    continue
                mentions = []
                for e in node["timelineItems"]["nodes"]:
                    s = (e or {}).get("source") or {}
                    if not e.get("isCrossRepository") or s.get("__typename") != "Issue" or s["repository"]["isPrivate"]:
                        continue
                    mentions.append({"repo": s["repository"]["nameWithOwner"], "fork": s["repository"]["isFork"], "number": s["number"], "title": s["title"],
                                     "text": (s.get("bodyText") or "")[:1600], "opened_at": s["createdAt"][:10], "state": s["state"], "url": s["url"],
                                     "author": (s.get("author") or {}).get("login"), "author_kind": (s.get("author") or {}).get("__typename"),
                                     "mentioned_at": e["referencedAt"][:10]})
                if mentions:
                    found[f"ISSUE-{node['number']}"] = mentions
    got = {"repo": repo, "read_at": datetime.date.today().isoformat(), "bug_reports_checked": len(numbers), "mentioned": found}
    LINKS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got))
    return got


REFERENCE = re.compile(r"https?://\S+|[\w.-]+/[\w.-]+#\d+|(?<![\w/])#\d+|\bGH-\d+", re.I)
NOT_A_REPORT = re.compile(r"dependency dashboard|^renovate|^(chore|build|ci|deps?|release)(\(.+?\))?[:!]|^(bump|update|upgrade)s?\b.*\b(from|to)\b\s+v?\d|"
                          r"^release\b|changelog|release notes|^\[?(weekly|monthly|automated)|^tracking\b|roadmap", re.I)
BOT = re.compile(r"\[bot\]$|^(renovate|dependabot|github-actions|mend|snyk|pyup|depfu|greenkeeper)", re.I)
RECORDS, REPORTS = bm.TOOL_RECORDS, bm.TOOL_REPORTS          # read as the hosted tool reads: 30 records from 120 per half, 20 reports
PRODUCT, WORDS, MEANING, RECENT = "Diastella's search", "keywords alone", "meaning alone", "rule: the most recently reported bugs"


def clean(text: str) -> str:
    """A reporter's words without the links and issue numbers that would give the answer away."""
    return re.sub(r"\s+", " ", REFERENCE.sub(" ", text or "")).strip()


def words_of(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2}


def pairs() -> list[dict]:
    """One row per downstream report that points at an upstream bug which was already reported when it was written."""
    rows = []
    for repo in bm.REPOS:
        got, data = links(repo), bm.fetch(repo)
        if not got or not data:
            continue
        bugs = bm.prepare(data)
        title = {f"ISSUE-{i['number']}": i["title"] for i in data["incidents"]}
        by_question: dict[tuple, dict] = {}
        for bid, mentions in got["mentioned"].items():
            for d in mentions:
                t = bm.day(d["opened_at"])
                if (d["repo"].lower() == repo.lower() or (d["fork"] and d["repo"].split("/")[1].lower() == repo.split("/")[1].lower())
                        or d["author_kind"] == "Bot" or BOT.search(d["author"] or "") or NOT_A_REPORT.search(d["title"]) or len(clean(d["title"])) < 12
                        or bid not in bugs or bugs[bid]["day"] >= t):
                    continue
                row = by_question.setdefault((d["repo"], d["number"]), {
                    "upstream": repo, "downstream": f"{d['repo']}#{d['number']}", "url": d["url"], "day": t, "title": clean(d["title"]),
                    "report": bm.as_the_tool_cleans(clean(d["title"] + "\n" + d["text"]), bm.TOOL_QUESTION), "answers": [], "fixed_before": False, "overlap": 0.0,
                    "opened_at": d["opened_at"], "linked_later": True})
                row["answers"].append(bid)
                row["linked_later"] = row["linked_later"] and d["mentioned_at"] > d["opened_at"]
                row["fixed_before"] = row["fixed_before"] or any(f["day"] < t for f in bugs[bid]["fixes"])
                a, b = words_of(row["title"]), words_of(title[bid])
                row["overlap"] = max(row["overlap"], len(a & b) / max(1, len(a | b)))
                if clean(title[bid]).lower()[:60] in row["report"].lower():        # the report quotes the upstream title: not a fair long question
                    row["report"] = None
        for row in by_question.values():
            row["known_bugs_then"] = sum(1 for b in bugs.values() if b["day"] < row["day"])
            row["recent"] = sorted((b for b in bugs if bugs[b]["day"] < row["day"]), key=lambda b: -bugs[b]["day"])[:REPORTS]
            rows.append(row)
    return rows


def ask(repo: str | None, text: str, t: int) -> dict[str, list[tuple[str, str]]]:
    """Upstream bug reports, closest first, as (project, report). With no project named, every stored project is searched."""
    from qdrant_client import models
    c = recall.client()
    must = [models.FieldCondition(key="day", range=models.Range(lt=t))]
    if repo:
        must.append(models.FieldCondition(key="repo", match=models.MatchValue(value="bench/" + repo)))
    flt, sparse, dense = models.Filter(must=must), recall._sparse(text), recall._dense(text)

    def reports(points) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for p in points:
            out += [(p.payload["repo"][6:], r) for r in p.payload["reports"] if (p.payload["repo"][6:], r) not in out]
        return out[:REPORTS]
    fused = c.query_points(bm.COLLECTION, limit=RECORDS, with_payload=["repo", "reports"], query=models.FusionQuery(fusion=models.Fusion.RRF),
                           prefetch=[models.Prefetch(query=sparse, using="bm25", filter=flt, limit=RECORDS * 4),
                                     models.Prefetch(query=dense, using="dense", filter=flt, limit=RECORDS * 4)]).points
    words = c.query_points(bm.COLLECTION, query=sparse, using="bm25", query_filter=flt, limit=RECORDS, with_payload=["repo", "reports"]).points
    meaning = c.query_points(bm.COLLECTION, query=dense, using="dense", query_filter=flt, limit=RECORDS, with_payload=["repo", "reports"]).points
    return {PRODUCT: reports(fused), WORDS: reports(words), MEANING: reports(meaning)}


GITHUB = LINKS / "github_search.json"
MODES = {"GitHub issue search, keywords": None, "GitHub issue search, by meaning": "semantic", "GitHub issue search, both": "hybrid"}
STOP = set("the a an and or of to in on for with when is are was be not no it its this that from by as at into after before if then than can cannot "
           "does do did using use used via how why what which while during have has had will would should could get got".split())


EQUAL = "GitHub issue search, both, bug-labelled reports only"     # the same haystack as Diastella's: asked after the other three
_LABELS: dict[str, str | None] = {}


def _label(repo: str) -> str | None:
    if repo not in _LABELS:
        _LABELS[repo] = bm.gh._bug_label(repo)
    return _LABELS[repo]


def _github(repo: str, before: str, text: str, mode: str | None, label: str | None = None) -> list[int] | None:
    """The first ten issues GitHub's own search returns in the upstream project for these words, among issues opened earlier."""
    words = re.sub(r"[^\w .\-]+", " ", text)[:200].strip()
    args = ["gh", "api", "-X", "GET", "search/issues", "-f", f"q=repo:{repo} is:issue created:<{before} " + (f'label:"{label}" ' if label else "") + words, "-f", "per_page=10"]
    for attempt in range(6):
        out = subprocess.run(args + (["-f", f"search_type={mode}"] if mode else []), capture_output=True, text=True)
        try:
            got = json.loads(out.stdout or "{}")
        except json.JSONDecodeError:
            got = {}
        if "items" in got:
            return [i["number"] for i in got["items"]]
        if "rate limit" not in (got.get("message", "") + out.stderr).lower() and "secondary" not in (got.get("message", "") + out.stderr).lower():
            return None                                                    # a question the search refuses (too long, odd characters)
        time.sleep(25)
    return None


def github(sample: int = 200) -> dict:
    """Ask GitHub's issue search the same title questions, for a fixed random sample of the pairs. Kept on disk, so a
    run that stops continues. GitHub searches every issue of the project as it reads today, comments included; the
    keyword search gets a second try with the five longest content words when the whole title finds nothing right."""
    rows = pairs()
    random.Random(20261117).shuffle(rows)                                  # the pairs whose writer had not yet named the upstream bug go first:
    chosen = ([r for r in rows if r["linked_later"]] + [r for r in rows if not r["linked_later"]])[:sample]   # they are the fair test
    done = json.loads(GITHUB.read_text()) if GITHUB.exists() else {}
    for n, row in enumerate(chosen):
        one = done.setdefault(row["downstream"], {})
        answers = {int(a.split("-")[1]) for a in row["answers"]}
        for name, mode in MODES.items():
            if name in one:
                continue
            found = _github(row["upstream"], row["opened_at"], row["title"], mode)
            time.sleep(2.1)
            if mode is None and not (found and answers & set(found)):          # an agent would shorten a keyword query that found nothing
                short = " ".join(sorted({w for w in re.findall(r"[A-Za-z0-9_.\-]{3,}", row["title"]) if w.lower() not in STOP}, key=len, reverse=True)[:5])
                again = _github(row["upstream"], row["opened_at"], short, None) if short else None
                time.sleep(2.1)
                found = (found or []) + [x for x in (again or []) if x not in (found or [])]
            one[name] = next((k + 1 for k, x in enumerate(found or []) if x in answers), None) if found is not None else "refused"
        if EQUAL not in one and _label(row["upstream"]):
            found = _github(row["upstream"], row["opened_at"], row["title"], "hybrid", _label(row["upstream"]))
            time.sleep(6.2)                                                # semantic and hybrid searches: ten a minute
            one[EQUAL] = next((k + 1 for k, x in enumerate(found or []) if x in answers), None) if found is not None else "refused"
        GITHUB.write_text(json.dumps(done))
        if n % 20 == 19:
            print(f"  {n + 1} of {len(chosen)} questions put to GitHub's search", flush=True)
    return done


def against_github(rows: list[dict]) -> dict | None:
    if not GITHUB.exists():
        return None
    asked = {k: v for k, v in json.loads(GITHUB.read_text()).items() if all(m in v for m in MODES)}
    ours_name = "Diastella's search"
    hit = lambda rank, k: bool(isinstance(rank, int) and rank <= k)   # noqa: E731

    def compare(both: list[dict]) -> dict | None:
        if not both:
            return None
        mine = [r["ranks"]["title"][PRODUCT] for r in both]
        table = {ours_name: {"5": round(sum(hit(x, 5) for x in mine) / len(both), 4), "10": round(sum(hit(x, 10) for x in mine) / len(both), 4)}}
        rng, versus = random.Random(20261117), {}
        for mode in MODES:
            theirs = [asked[r["downstream"]][mode] for r in both]
            table[mode] = {"5": round(sum(hit(x, 5) for x in theirs) / len(both), 4), "10": round(sum(hit(x, 10) for x in theirs) / len(both), 4),
                           "refused": sum(1 for x in theirs if x == "refused")}
            a, b = [hit(x, 5) for x in mine], [hit(x, 5) for x in theirs]
            versus[mode] = {"difference_in_the_first_5": round((sum(a) - sum(b)) / len(a), 4), **bm.mcnemar(a, b),
                            "interval_95": bm.interval([float(x) for x in a], [float(x) for x in b], rng),
                            "both_found_it": sum(1 for x, y in zip(a, b) if x and y), "only_diastella_found_it": sum(1 for x, y in zip(a, b) if x and not y),
                            "only_github_found_it": sum(1 for x, y in zip(a, b) if y and not x), "neither": sum(1 for x, y in zip(a, b) if not x and not y)}
        same = [r for r in both if EQUAL in asked[r["downstream"]]]
        if same:
            mine_same, theirs = [r["ranks"]["title"][PRODUCT] for r in same], [asked[r["downstream"]][EQUAL] for r in same]
            a, b = [hit(x, 5) for x in mine_same], [hit(x, 5) for x in theirs]
            versus[EQUAL] = {"questions": len(same), "diastella_in_the_first_5": round(sum(a) / len(a), 4), "github_in_the_first_5": round(sum(b) / len(b), 4),
                             "difference_in_the_first_5": round((sum(a) - sum(b)) / len(a), 4), **bm.mcnemar(a, b),
                             "interval_95": bm.interval([float(x) for x in a], [float(x) for x in b], rng),
                             "only_diastella_found_it": sum(1 for x, y in zip(a, b) if x and not y), "only_github_found_it": sum(1 for x, y in zip(a, b) if y and not x)}
        best = [min((x for x in (asked[r["downstream"]][m] for m in MODES) if isinstance(x, int)), default=None) for r in both]
        table["GitHub, the best of its three searches for each question"] = {"5": round(sum(hit(x, 5) for x in best) / len(both), 4),
                                                                             "10": round(sum(hit(x, 10) for x in best) / len(both), 4)}
        return {"questions": len(both), "the_linked_upstream_bug_was_in_the_first": table, "diastella_against_each": versus}
    done = [r for r in rows if r["downstream"] in asked]
    return {"the_writer_had_not_yet_named_the_upstream_bug": compare([r for r in done if r["linked_later"]]), "every_pair_asked_so_far": compare(done),
            "note": "The same title, the same upstream project, only issues opened before the downstream report. GitHub searches every issue of the "
                    "project as it reads today, comments included (which can contain the downstream link itself); Diastella searches the "
                    "project's bug-labelled reports, text and labels as they read today, cut to those opened before the downstream report. "
                    "GitHub's keyword search is given a second try with fewer words."}


def place(ranked: list[tuple[str, str]], repo: str, answers: list[str]) -> int | None:
    return next((k + 1 for k, r in enumerate(ranked) if r[0].lower() == repo.lower() and r[1] in answers), None)


def replay(rows: list[dict]) -> None:
    for n, row in enumerate(rows):
        row["ranks"] = {}
        for way in ("title", "report"):
            text = row[way]
            if not text:
                row["ranks"][way] = None
                continue
            inside, anywhere = ask(row["upstream"], text, row["day"]), ask(None, text, row["day"])
            row["ranks"][way] = {**{m: place(r, row["upstream"], row["answers"]) for m, r in inside.items()},
                                 **{m + ", project not named": place(r, row["upstream"], row["answers"]) for m, r in anywhere.items()},
                                 RECENT: place([(row["upstream"], b) for b in row["recent"]], row["upstream"], row["answers"])}
        if n % 200 == 199:
            print(f"  {n + 1} of {len(rows)} questions asked", flush=True)


def summarise(rows: list[dict], way: str) -> dict:
    rows = [r for r in rows if r["ranks"].get(way)]
    if not rows:
        return {}
    methods, rng, rng2 = list(rows[0]["ranks"][way]), random.Random(20261117), random.Random(20261118)
    hit = lambda r, m, k: bool(r["ranks"][way][m] and r["ranks"][way][m] <= k)   # noqa: E731
    rate = lambda rs, m, k: round(sum(hit(r, m, k) for r in rs) / len(rs), 4)   # noqa: E731
    project = [r["upstream"] for r in rows]
    versus = {}
    for other in [m for m in methods if m != PRODUCT]:
        a, b = [hit(r, PRODUCT, 5) for r in rows], [hit(r, other, 5) for r in rows]
        versus[other] = {"difference": round((sum(a) - sum(b)) / len(a), 4), **bm.mcnemar(a, b), "interval_95": bm.interval([float(x) for x in a], [float(x) for x in b], rng),
                         "interval_95_by_project": bm.interval_by_project([float(x) for x in a], [float(x) for x in b], project, rng2)}
    chance = round(sum(min(1.0, 5 / max(1, r["known_bugs_then"])) for r in rows) / len(rows), 4)
    return {"questions": len(rows), "upstream_projects": len(set(project)), "downstream_projects": len({r["downstream"].split("#")[0] for r in rows}),
            "found_in_the_first": {m: {"1": rate(rows, m, 1), "5": rate(rows, m, 5), "10": rate(rows, m, 10)} for m in methods},
            "by_chance_in_the_first_5": chance, "product_against_each_other_in_the_first_5": versus,
            "per_upstream_project_first_5": {rp: {"questions": sum(1 for r in rows if r["upstream"] == rp),
                                                  **{m: rate([r for r in rows if r["upstream"] == rp], m, 5) for m in (PRODUCT, WORDS, MEANING, RECENT)}}
                                             for rp in sorted(set(project))}}


LIMITS = [
    "The links are the ones GitHub records when a report or a comment in one repository mentions an issue in another. A mention is not proof "
    "that the upstream bug was the cause: some are 'possibly related'. Nobody read the pairs to remove those.",
    "Only pairs where the upstream bug had already been reported when the downstream report was written are replayed, since only then "
    "could a memory have answered. The upstream memory is cut to the day before the downstream report.",
    "The question is the downstream reporter's title with links and issue numbers removed. The longer question (title and start of the "
    "text, cleaned and cut to 600 characters as the hosted tool does) leaves out reports that quote the upstream bug's title.",
    "Reports opened by bots, dependency-update notices and release notes are removed by rules on the title and author; some will have "
    "slipped through and some real reports will have been removed.",
    "The upstream project is told to the search except in the rows marked 'project not named', where all 22 stored projects are searched "
    "at once. In real use the candidates would be the project's own dependencies, which this test does not model.",
    "The search reads as the hosted tool reads: 30 records from 120 per half, the first 20 reports. Only bug-labelled upstream reports "
    "are stored, at most 480 per project, so an upstream bug outside that window cannot be an answer and is not counted.",
    "No language model and no agent is involved; the only model is the embedding model behind search by meaning (all-MiniLM-L6-v2).",
]

# What a separate check of 17 November 2026 found. The checker was automated: an AI reviewer that did not write this
# script recomputed the figures from the cached answers and re-asked GitHub 27 times. No outside person has reviewed
# the work yet. Kept with the result so the numbers are never quoted without it.
REVIEW = {
    "reviewed_on": "2026-11-17",
    "reviewed_by": "an automated reviewer (an AI agent that did not write the benchmark); no outside person yet",
    "figures_recomputed_and_matching": True,
    "wording_that_may_be_published": (
        "In 297 cases where a report filed in one project was later linked to a bug already reported in a project it uses, and the link "
        "was made on a later day than the report, Diastella's search over that project's recent bug-labelled reports (at most 480) put the "
        "linked bug in its first five results for 42%, from the report's title alone. GitHub's hybrid issue search, given the same title "
        "once and limited to issues with the same bug label, did so for 22% (15% across all of the project's issues). A link is not proof "
        "of cause, and these are only cases someone eventually linked."),
    "what_must_be_said_with_the_numbers": [
        "The pools differ. Diastella holds at most the 480 most recent bug-labelled reports of a project, and the answer is always among "
        "them; GitHub with the label searches every labelled issue opened before the date (thousands in large projects), and without it "
        "every issue. On the 65 questions from the 7 projects where both searched exactly the same reports: Diastella 49%, GitHub 31%.",
        "Most of the lead comes from the small prepared pool, not from the way the two rankings are fused: keywords alone score 38% and "
        "meaning alone 36% on the same questions.",
        "A mention is not a cause. Of 15 pairs read by the reviewer, 12 were a real downstream symptom with an upstream cause, 2 were a "
        "commenter's guess and 1 a passing mention.",
        "The upstream project is not always a library the downstream project depends on: at least 81 of the 297 are servers or tools.",
        "In 11 of the 297 the downstream report as it reads today already carries the upstream reference (issues that were moved or "
        "imported). Without them: 41%, 22% and 15%.",
        "GitHub's search was given the title word for word, once. It often returns very few results for such a question; a person would "
        "rephrase. Only its keyword search was given a second try.",
        "This measures how often a link people did make would have been found. It says nothing about how often a failure has a known "
        "upstream cause, or about false alarms when it has none.",
    ],
}

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    recall.COLLECTION = bm.COLLECTION
    if what == "links":
        for repo in sys.argv[2:] or bm.REPOS:
            started = time.time()
            got = links(repo)
            if got:
                print(f"  {repo}: {got['bug_reports_checked']} bug reports checked, {len(got['mentioned'])} mentioned by {sum(len(m) for m in got['mentioned'].values())} "
                      f"issues elsewhere ({time.time() - started:.0f}s)", flush=True)
        sys.exit(0)
    rows = pairs()
    print(f"{len(rows)} downstream reports point at an upstream bug that was already reported", flush=True)
    if what == "pairs":
        for r in random.Random(7).sample(rows, min(25, len(rows))):
            print(f"  {r['downstream']:<44} {r['title'][:70]!r}  ->  {r['upstream']} {r['answers']}  fixed_before={r['fixed_before']} overlap={r['overlap']:.2f}")
        sys.exit(0)
    if what == "github":
        github(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
        sys.exit(0)
    kept = LINKS / "rows.json"
    before = {r["downstream"]: r for r in json.loads(kept.read_text())} if kept.exists() and "fresh" not in sys.argv else {}
    if all(r["downstream"] in before and before[r["downstream"]]["answers"] == r["answers"] for r in rows):
        for r in rows:                                          # the same questions as last time: where each search put the answer is on disk
            was = before[r["downstream"]]
            r["ranks"] = was["ranks"] if "ranks" in was else {"title": was["title"], "report": was["report"]}
    else:
        replay(rows)
    subsets = {"every pair": rows, "the upstream bug was already fixed": [r for r in rows if r["fixed_before"]],
               "the link was made on a later day than the report (its writer had not yet named the upstream bug)": [r for r in rows if r["linked_later"]],
               "worded differently from the upstream report (under half the title words shared)": [r for r in rows if r["overlap"] < 0.5]}
    versus_github = against_github(rows)
    checked = sum((links(repo) or {}).get("bug_reports_checked", 0) for repo in bm.REPOS)
    mentioned = sum(len((links(repo) or {}).get("mentioned", {})) for repo in bm.REPOS)
    result = {"run_on": datetime.date.today().isoformat(), "method": __doc__.split("\n\n")[2].replace("\n", " "), "limits": LIMITS,
              "links": {"upstream_bug_reports_checked": checked, "mentioned_by_an_issue_in_another_repository": mentioned, "questions_replayed": len(rows)},
              "asked_with_the_title": {name: summarise(rs, "title") for name, rs in subsets.items()},
              "asked_with_the_start_of_the_report": {name: summarise(rs, "report") for name, rs in subsets.items()},
              **({"against_githubs_own_search": versus_github, "separate_check": REVIEW} if versus_github else {})}
    bm.OUT.mkdir(parents=True, exist_ok=True)
    (bm.OUT / f"upstream-{result['run_on']}.json").write_text(json.dumps(result, indent=1))
    (LINKS / "rows.json").write_text(json.dumps(rows))
    show = lambda block: {name: {"questions": v.get("questions"), "found_in_the_first": v.get("found_in_the_first"), "chance": v.get("by_chance_in_the_first_5")} for name, v in block.items()}   # noqa: E731
    print(json.dumps({"links": result["links"], "title": show(result["asked_with_the_title"]), "report": show(result["asked_with_the_start_of_the_report"]),
                      "github": versus_github}, indent=1))
