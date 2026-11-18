"""Does a project's failure memory point at the file a real fix changed?

A replay with no language model and no agent in it; the only model is the small embedding model behind the search by
meaning. For every bug-labelled report of a public project that was closed by a merged change and passes the filters
listed under "limits", the memory is cut back to the day the bug was reported (the bug itself, its fix and everything
later removed), the bug's own words are used as the question, and each way of searching the memory returns earlier
fixed bugs. A search "hits" when a source-code file changed by one of the fixes it returned is a file the real fix of
this bug changed. Every bug is asked twice: with its report (the title and the start of the text, up to 1,600
characters), and with its title alone, which stands in for a short question.

Compared: the product's search (meaning and words fused, the same fusion as the hosted service's search (diastella_bench/search.py) but read deeper: 80
records from 320 per half), the same search read as the hosted tool for coding agents reads it (the question cleaned
and cut to 600 characters, 30 records from 120 per half, the first 20 reports), each half alone, and three rules that
need no search at all (the files fixed most often, the most recent fixes, random earlier fixes). The ceiling is the
share of bugs for which ANY earlier fix touched one of the files.

    uv run python benchmark_memory.py fetch      # read the projects from GitHub (cached under .cache/)
    uv run python benchmark_memory.py run        # index locally, replay, write the results

Needs a local Qdrant (docker compose up -d) and fastembed; touches only its own collection. Results go to
results/memory-<date>.json. Limits are listed in the output under "limits" and must be quoted with the numbers.
Whether words should count for less than meaning in the product's search was tested separately (tune_search.py); its
outcome is copied into the result as "weighting_check"."""
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")                                      # QDRANT_URL of the local search service
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from diastella_bench import reader as gh  # noqa: E402
from diastella_bench import search as recall  # noqa: E402
from diastella_bench.clean import quoted as as_the_tool_cleans  # noqa: E402  (how the hosted tool cleans a question)

CACHE = ROOT / ".cache" / "memory"
OUT = ROOT / "results"
COLLECTION = "memory_benchmark"
REPOS = ["kornia/kornia", "pydantic/pydantic", "celery/celery", "caddyserver/caddy", "PrefectHQ/prefect", "streamlit/streamlit",
         "tokio-rs/tokio", "vitejs/vite", "kubeflow/pipelines", "gradio-app/gradio", "mlflow/mlflow", "argoproj/argo-cd",
         "scrapy/scrapy", "pytest-dev/pytest", "sveltejs/svelte", "encode/httpx", "open-telemetry/opentelemetry-demo",
         "dapr/dapr", "grafana/k6", "traefik/traefik", "cli/cli", "prometheus/alertmanager", "fastapi/fastapi", "nestjs/nest"]
MIN_EARLIER = 20          # a bug is replayed only if at least this many earlier bugs already had a merged fix
MAX_TRUTH = 8             # and its own fix changed at most this many source files (a sweeping change names no place)
DEPTH = 80                # how many records each search returns before they are turned into earlier bugs
NO_SEARCH = ("most recent fixes", "files fixed most often", "random earlier fixes")     # rules that never read the question
PRODUCT, AS_THE_TOOL = "product search (meaning + words)", "product search, read as the hosted tool reads it"
TOOL_QUESTION, TOOL_RECORDS, TOOL_REPORTS = 600, 30, 20   # the hosted tool cuts a question to 600 characters and asks for 20 reports;
#                                                           the search then returns at most 30 records, read from 120 per half
SELECTION: dict = {"bug_reports_read": 0, "with_a_merged_fix": 0, "replayed": 0, "left_out": {},
                   "replayed_but_imperfect": {"the fix is a change in another repository": 0, "the report is open today": 0}}
NOT_SOURCE = re.compile(r"(^|/)(tests?|testing|__tests__|spec|specs|docs?|documentation|examples?|samples?|benchmarks?|fixtures?|testdata|"
                        r"e2e|snapshots?|__snapshots__|vendor|third_party|node_modules|dist|build|\.github|changelog\.d|changes|newsfragments)(/|$)", re.I)
NOT_SOURCE_FILE = re.compile(r"(^|/)(changelog|changes|history|news|readme|contributing|authors|license|notice|codeowners)(\.[a-z]+)?$|"
                             r"(_test|\.test|\.spec|_spec|\.snap|\.lock|\.min)\.[a-z0-9]+$|(^|/)test_[^/]+$|\.(md|rst|txt|lock|sum|svg|png|jpg|json|ya?ml|toml|cfg|ini)$", re.I)


def is_source(path: str) -> bool:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    return ext in gh.CODE and not NOT_SOURCE.search(path) and not NOT_SOURCE_FILE.search(path)


def day(text: str) -> int:
    return int(text.replace("-", ""))


def fetch(repo: str) -> dict | None:
    path = CACHE / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    meta = gh._api(f"repos/{repo}")
    label = gh._bug_label(repo)
    if not meta or not label:
        print(f"  {repo}: no bug label, skipped"); return None
    services, _, _ = gh._systems(repo, meta["default_branch"])
    data = {"repo": repo, "fetched_at": datetime.date.today().isoformat(), "connector_version": gh.CONNECTOR_VERSION,
            "incidents": gh._incidents(repo, label, services, 12)}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def prepare(data: dict) -> dict:
    """Per project: for each bug the day it was reported, and its fixes with their day and the source files they changed."""
    bugs = {}
    for i in data["incidents"]:
        fixes = [{"day": day(f["merged_at"]), "files": [p for p in f.get("files") or [] if is_source(p)],
                  "complete": (f.get("changed_files") or 99) <= 12} for f in i["fixes"]]
        bugs[f"ISSUE-{i['number']}"] = {"day": day(i["opened_at"]), "text": f"{i['title']}\n{i.get('text') or ''}", "fixes": fixes}
    return bugs


def files_as_of(bug: dict, t: int) -> list[str]:
    """The source files changed by this earlier bug's fixes that had been merged before day t."""
    out: list[str] = []
    for f in bug["fixes"]:
        if f["day"] < t:
            out += [p for p in f["files"] if p not in out]
    return out


def index(repo: str, data: dict) -> str:
    """Store the project's records exactly as the product does (same texts, same two vectors), plus the day as a number
    so that a search can be limited to what existed on a given day."""
    from qdrant_client import models
    key = "bench/" + repo
    snap = {**data, "repo": key, "private": False}
    c = recall.client()
    done = c.count(COLLECTION, count_filter=recall._repo_filter(key), exact=True).count if c.collection_exists(COLLECTION) else 0
    want = sum(1 for _ in recall._records(snap))
    if done == want:
        return key
    recall.ensure()
    c.delete(COLLECTION, points_selector=models.FilterSelector(filter=recall._repo_filter(key)))
    points = []
    for rid, text, payload in recall._records(snap):
        dense = recall._dense(text)
        points.append(models.PointStruct(id=hashlib.md5(f"{key}#{rid}".encode()).hexdigest(), payload={**payload, "day": day(payload["opened_at"])},
                                         vector={"bm25": recall._sparse(text), **({"dense": dense} if dense else {})}))
    for k in range(0, len(points), 48):
        c.upsert(COLLECTION, points=points[k:k + 48], wait=True)
    return key


def searches(key: str, text: str, t: int) -> dict[str, list[str]]:
    """Earlier bugs, closest first, by the product's fused search and by each half alone, among records dated before t."""
    from qdrant_client import models
    c = recall.client()
    flt = models.Filter(must=[models.FieldCondition(key="repo", match=models.MatchValue(value=key)),
                              models.FieldCondition(key="day", range=models.Range(lt=t))])
    sparse, dense = recall._sparse(text), recall._dense(text)

    def bugs(points) -> list[str]:
        out: list[str] = []
        for p in points:
            out += [r for r in p.payload["reports"] if r not in out]
        return out
    fused = c.query_points(COLLECTION, limit=DEPTH, with_payload=True, query=models.FusionQuery(fusion=models.Fusion.RRF),
                           prefetch=[models.Prefetch(query=sparse, using="bm25", filter=flt, limit=DEPTH * 4),
                                     models.Prefetch(query=dense, using="dense", filter=flt, limit=DEPTH * 4)]).points
    words = c.query_points(COLLECTION, query=sparse, using="bm25", query_filter=flt, limit=DEPTH, with_payload=True).points
    meaning = c.query_points(COLLECTION, query=dense, using="dense", query_filter=flt, limit=DEPTH, with_payload=True).points
    asked = as_the_tool_cleans(text, TOOL_QUESTION)
    tool = c.query_points(COLLECTION, limit=TOOL_RECORDS, with_payload=True, query=models.FusionQuery(fusion=models.Fusion.RRF),
                          prefetch=[models.Prefetch(query=recall._sparse(asked), using="bm25", filter=flt, limit=TOOL_RECORDS * 4),
                                    models.Prefetch(query=recall._dense(asked), using="dense", filter=flt, limit=TOOL_RECORDS * 4)]).points
    return {PRODUCT: bugs(fused), AS_THE_TOOL: bugs(tool)[:TOOL_REPORTS], "words only": bugs(words), "meaning only": bugs(meaning)}


def eligible(bugs: dict, left_out: dict | None = None):
    """The bugs that are replayed, oldest first: (id, bug, day reported, source files its real fix changed, the earlier
    bugs that already had a merged fix on that day with the files those fixes had changed). `left_out` counts the fixed
    bugs that are not replayed, by reason."""
    def skip(why: str) -> None:
        if left_out is not None:
            left_out[why] = left_out.get(why, 0) + 1
    for bid, bug in sorted(bugs.items(), key=lambda kv: kv[1]["day"]):
        t = bug["day"]
        truth = {p for f in bug["fixes"] for p in f["files"]}
        if not bug["fixes"]:
            continue
        if not all(f["complete"] for f in bug["fixes"]):
            skip("its fix changed more than 12 files"); continue
        if not truth:
            skip("its fix changed no source-code file (only tests, documentation or configuration)"); continue
        if len(truth) > MAX_TRUTH:
            skip(f"its fix changed more than {MAX_TRUTH} source-code files"); continue
        earlier = {eid: files_as_of(e, t) for eid, e in bugs.items() if eid != bid and e["day"] < t}
        earlier = {eid: fs for eid, fs in earlier.items() if fs}               # earlier bugs that already had a merged fix
        if len(earlier) < MIN_EARLIER:
            skip(f"fewer than {MIN_EARLIER} earlier bugs had a merged fix when it was reported"); continue
        yield bid, bug, t, truth, earlier


def score(ranked: list[str], earlier: dict[str, list[str]], truth: set[str]) -> dict:
    ranked = [r for r in ranked if r in earlier]                                # never the bug itself, never a later one
    files: list[str] = []
    for r in ranked:
        files += [p for p in earlier[r] if p not in files]
    first = next((k + 1 for k, r in enumerate(ranked) if truth & set(earlier[r])), None)
    return {"bug@1": first is not None and first <= 1, "bug@3": first is not None and first <= 3, "bug@5": first is not None and first <= 5,
            "file@3": bool(truth & set(files[:3])), "file@5": bool(truth & set(files[:5])), "file@10": bool(truth & set(files[:10]))}


def replay(repo: str, data: dict, rng: random.Random) -> list[dict]:
    bugs = prepare(data)
    key = index(repo, data)
    rows = []
    SELECTION["bug_reports_read"] += len(bugs)
    SELECTION["with_a_merged_fix"] += sum(1 for b in bugs.values() if b["fixes"])
    for bid, bug, t, truth, earlier in eligible(bugs, SELECTION["left_out"]):
        out = {name: score(ranked, earlier, truth) for name, ranked in searches(key, bug["text"], t).items()}
        by_recency = sorted(earlier, key=lambda e: -max(f["day"] for f in bugs[e]["fixes"] if f["day"] < t))
        out["most recent fixes"] = score(by_recency, earlier, truth)
        count: dict[str, int] = {}
        for fs in earlier.values():
            for p in fs:
                count[p] = count.get(p, 0) + 1
        often = sorted(count, key=lambda p: (-count[p], p))
        out["files fixed most often"] = {"bug@1": None, "bug@3": None, "bug@5": None, "file@3": bool(truth & set(often[:3])),
                                         "file@5": bool(truth & set(often[:5])), "file@10": bool(truth & set(often[:10]))}
        draws = []
        for _ in range(20):
            order = list(earlier); rng.shuffle(order); draws.append(score(order, earlier, truth))
        out["random earlier fixes"] = {m: sum(d[m] for d in draws) / len(draws) for m in draws[0]}
        short = {name: score(ranked, earlier, truth) for name, ranked in searches(key, bug["text"].split("\n", 1)[0], t).items()}
        rows.append({"repo": repo, "bug": bid, "day": t, "truth_files": len(truth), "earlier_fixed_bugs": len(earlier),
                     "possible": any(truth & set(fs) for fs in earlier.values()), "methods": out,
                     "title_alone": {**short, **{m: out[m] for m in NO_SEARCH}}})
    SELECTION["replayed"] += len(rows)
    by, flaws = {f"ISSUE-{i['number']}": i for i in data["incidents"]}, SELECTION["replayed_but_imperfect"]
    for row in rows:
        i = by[row["bug"]]
        flaws["the fix is a change in another repository"] += any(f"github.com/{repo}/".lower() not in (f.get("url") or "").lower() for f in i["fixes"])
        flaws["the report is open today"] += (i.get("status") or "").lower() != "closed"
    return rows


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided test on the bugs where exactly one of two methods hit."""
    only_a, only_b = sum(1 for x, y in zip(a, b) if x and not y), sum(1 for x, y in zip(a, b) if y and not x)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": round(p, 4) if p >= 0.0001 else "<0.0001"}


def interval(a: list[float], b: list[float], rng: random.Random, n: int = 2000) -> list[float]:
    idx, diffs = range(len(a)), []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        diffs.append(sum(a[i] - b[i] for i in pick) / len(pick))
    diffs.sort()
    return [round(diffs[int(0.025 * n)], 4), round(diffs[int(0.975 * n)], 4)]


def interval_by_project(a: list[float], b: list[float], project: list[str], rng: random.Random, n: int = 2000) -> list[float]:
    """The same interval with whole projects drawn instead of single bugs. Bugs of one project resemble each other, so
    this is the stricter reading: a difference that holds here does not depend on a few large projects."""
    by: dict[str, list[float]] = {}
    for x, y, repo in zip(a, b, project):
        one = by.setdefault(repo, [0.0, 0.0])
        one[0] += x - y; one[1] += 1
    groups, diffs = list(by.values()), []
    for _ in range(n):
        pick = [rng.choice(groups) for _ in groups]
        diffs.append(sum(g[0] for g in pick) / sum(g[1] for g in pick))
    diffs.sort()
    return [round(diffs[int(0.025 * n)], 4), round(diffs[int(0.975 * n)], 4)]


def summarise(rows: list[dict], asked: str = "methods") -> dict:
    """The figures for one way of asking: "methods" is the report (title and start of text), "title_alone" the title only."""
    rng, rng_projects = random.Random(20261117), random.Random(20261118)
    project = [r["repo"] for r in rows]

    def by_project(a: list[float], b: list[float]) -> dict:
        tally: dict[str, float] = {}
        for x, y, repo in zip(a, b, project):
            tally[repo] = tally.get(repo, 0.0) + x - y
        return {"interval_95_by_project": interval_by_project(a, b, project, rng_projects),
                "projects": {"ahead": sum(1 for v in tally.values() if v > 1e-9), "level": sum(1 for v in tally.values() if abs(v) <= 1e-9),
                             "behind": sum(1 for v in tally.values() if v < -1e-9)}}
    methods = list(rows[0][asked])
    metrics = ["file@3", "file@5", "file@10", "bug@1", "bug@3", "bug@5"]
    rate = lambda rs, m, k: (round(sum(float(r[asked][m][k]) for r in rs) / len(rs), 4) if rs[0][asked][m][k] is not None else None)   # noqa: E731
    repos = sorted({r["repo"] for r in rows})
    per_repo = {rp: {"bugs": sum(1 for r in rows if r["repo"] == rp),
                     **{m: rate([r for r in rows if r["repo"] == rp], m, "file@5") for m in methods}} for rp in repos}
    product, compare = PRODUCT, {}
    for other in [m for m in methods if m != PRODUCT]:
        a = [bool(r[asked][product]["file@5"]) for r in rows]
        if other == "random earlier fixes":
            b = [float(r[asked][other]["file@5"]) for r in rows]
            compare[other] = {"difference": round(sum(a) / len(a) - sum(b) / len(b), 4), "interval_95": interval([float(x) for x in a], b, rng),
                              **by_project([float(x) for x in a], b)}
            continue
        b = [bool(r[asked][other]["file@5"]) for r in rows]
        compare[other] = {"difference": round(sum(a) / len(a) - sum(b) / len(b), 4), **mcnemar(a, b),
                          "interval_95": interval([float(x) for x in a], [float(x) for x in b], rng),
                          **by_project([float(x) for x in a], [float(x) for x in b])}
    return {"bugs_replayed": len(rows), "projects": len(repos), "ceiling_any_earlier_fix_touched_a_right_file": round(sum(r["possible"] for r in rows) / len(rows), 4),
            "overall": {m: {k: rate(rows, m, k) for k in metrics} for m in methods},
            "average_of_projects_file@5": {m: round(sum(per_repo[rp][m] for rp in repos) / len(repos), 4) for m in methods},
            "product_search_against_each_other_method_file@5": compare, "per_project_file@5": per_repo}


LIMITS = [
    "The question put to the memory is the bug's own words: its title with the start of its text (up to 1,600 characters, as the report "
    "reads today), and separately its title alone. Neither is what a person or an agent in the middle of a failure would type.",
    "A hit means a returned fix changed one of the source-code files the real fix changed; it does not mean the bug would have been fixed. "
    "Tests, documentation and configuration files do not count, on either side.",
    "Not every fixed bug is replayed. Only reports that carry the project's bug label and were closed by a merged change are read, and of "
    "those the following are left out: bugs whose fix changed more than 12 files, no source-code file, or more than 8 of them, and bugs "
    "reported when fewer than 20 earlier bugs had a merged fix. The counts are in the result under \"selection\".",
    "At most 480 of a project's most recent bug reports are read, so a busy project is represented by its last months.",
    "Intervals and p values count each bug as independent. The stricter reading, with whole projects as the unit, is given beside them "
    "(interval_95_by_project): the lead over the simple rules and over keyword search holds there; the small differences between the "
    "combined search and search by meaning alone do not.",
    "The first row reads deeper than the product does: 80 records from 320 per half. The row \"read as the hosted tool reads it\" replays "
    "the search the way the address for coding agents makes it: the question cleaned and cut to 600 characters, 30 records from 120 per "
    "half, the first 20 reports.",
    "Figures move a little between runs. The search service does not fix the order of records whose scores are exactly equal, which changes "
    "the product's result for a few bugs in three thousand (the count against the run before is in the result under \"repeat_run\"). The "
    "word counts behind the keyword half change with the set of projects stored: the first run of 17 November 2026 stored the projects one "
    "at a time, and against it the product's result changed for about twenty bugs, up to two points in one small project, with no net change.",
    "History is cut by day, not by hour. How rare a word is, which the keyword half of the search relies on, is counted over every record "
    "stored (all of these projects, at any date), not only over what existed in that project on that day; the hosted service also keeps one "
    "count for all the projects it holds.",
    "The embedding model is all-MiniLM-L6-v2 run locally, the same model the hosted service uses. It reads about the first 256 word pieces "
    "of a text, so most of a long report is seen only by the keyword half.",
    "A small share of the replayed bugs are imperfect records: for some the fix is a change in another repository, and some are open "
    "today although a merged change is linked as closing them. Their counts are in the result under \"selection\" (about 3% together).",
    "No language model and no agent is involved: this measures what the memory contains and how well each search finds it, not what an "
    "agent does with it.",
]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    names = sys.argv[2:] or REPOS
    recall.COLLECTION = COLLECTION                              # the benchmark's own collection; the product's is never touched
    if what == "fetch":
        for repo in names:
            started = time.time()
            got = fetch(repo)
            if got:
                print(f"  {repo}: {len(got['incidents'])} bug reports, {sum(1 for i in got['incidents'] if i['fixes'])} with a merged fix ({time.time() - started:.0f}s)", flush=True)
        sys.exit(0)
    rng, rows = random.Random(20261117), []
    for repo in names:                                          # every project is stored before any is replayed, so that the word
        if fetch(repo):                                         # counts of the keyword half are the same for the first as for the last
            index(repo, fetch(repo))
    for repo in names:
        data = fetch(repo)
        if not data:
            continue
        started = time.time()
        got = replay(repo, data, rng)
        rows += got
        print(f"  {repo}: {len(got)} bugs replayed ({time.time() - started:.0f}s)", flush=True)
    product, short = PRODUCT, summarise(rows, "title_alone")
    result = {"run_on": datetime.date.today().isoformat(), "method": __doc__.split("\n\n")[1].replace("\n", " "), "limits": LIMITS,
              "selection": {"projects_read": len(names), "projects_with_a_replayed_bug": len({r["repo"] for r in rows}), **SELECTION},
              **summarise(rows),
              "asked_with_the_title_alone": {k: v for k, v in short.items() if k not in ("bugs_replayed", "projects", "ceiling_any_earlier_fix_touched_a_right_file")}}
    before = CACHE / "rows.json"
    if before.exists():                                         # how much two runs of the same search differ (order among equal scores)
        was = {(r["repo"], r["bug"]): r for r in json.loads(before.read_text())}
        both = [r for r in rows if (r["repo"], r["bug"]) in was]
        differs = lambda asked: sum(1 for r in both if asked in was[(r["repo"], r["bug"])]   # noqa: E731
                                    and r[asked][product]["file@5"] != was[(r["repo"], r["bug"])][asked][product]["file@5"])
        result["repeat_run"] = {"what": "bugs for which the product's search hit in one run and missed in the other (right file in the first five)",
                                "bugs_in_both_runs": len(both), "product_search_differs_on": differs("methods"),
                                "product_search_differs_on_title_alone": differs("title_alone")}
    weighting = sorted(OUT.glob("memory-weighting-*.json"))
    if weighting:
        w = json.loads(weighting[-1].read_text())
        names = {"search as it was (words and meaning equal)": "search as it is", "chosen weighting": "chosen weighting", "meaning alone": "meaning alone"}
        result["weighting_check"] = {
            "question": "Should the ranking by words count for less than the ranking by meaning?", "run_on": w["run_on"],
            "chosen_on": {"projects": len(w["chosen_on"]["projects"]), "bugs": w["chosen_on"]["bugs"]}, "chosen": w["chosen"],
            "held_out": {"projects": len(w["held_out"]["projects"]), "bugs": w["held_out"]["bugs"]},
            "held_out_file@5": {way: {names[n]: v["file@5"] for n, v in vals.items()} for way, vals in w["held_out"]["results"].items()},
            "chosen_against_search_as_it_is": {way: {k: v[k] for k in ("difference", "interval_95") if k in v} for way, v in w["held_out"]["chosen_against_search_as_it_was_file@5"].items()},
            "tied_when_chosen": w.get("tied_on_the_choosing_half"), "other_tied_setting_against_search_as_it_is": w.get("held_out", {}).get("other_tied_settings_against_search_as_it_was_file@5"),
            "outcome": w["decision"].get("outcome"), "search_changed": w["decision"]["change_the_search"],
            "note": "The search as it is was scored here from answers recorded a little earlier than the table above; one bug differs (order among equal scores).",
            "full_result": weighting[-1].name}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"memory-{result['run_on']}.json").write_text(json.dumps(result, indent=1))
    (CACHE / "rows.json").write_text(json.dumps(rows))
    short_view = {k: v for k, v in result["asked_with_the_title_alone"].items() if k != "per_project_file@5"}
    print(json.dumps({**{k: v for k, v in result.items() if k not in ("per_project_file@5", "limits", "method")}, "asked_with_the_title_alone": short_view}, indent=1))
