"""How much should words count against meaning when a project's failure memory is searched?

The first run of the memory benchmark (benchmark_memory.py) found that the product's search, which gave the ranking by
words the same say as the ranking by meaning, scored slightly below meaning alone. This script chooses a weighting and
then checks it on projects that had no part in the choice.

Written down before any weighting was scored:
- The projects are put in alphabetical order and every second one is held out. Settings are chosen on the others only.
- Each bug is asked two ways: with its full report (title and text), and with its title alone, which stands in for
  the short question a person or a coding agent usually types.
- The setting chosen is the one with the highest share of bugs, averaged over the two ways of asking, for which a file
  the real fix changed is among the first five returned (the benchmark's headline measure). Between equal settings:
  the one nearest the search as it was (words counting more, then the smaller constant).
- The search is changed only if, on the held-out projects, the chosen setting is better than the search as it was by
  that same average and not more than half a point worse on either way of asking. Whatever the held-out figures are,
  they are the ones reported.
One thing was known beforehand and is not hidden: the first run's figures per project for meaning alone and for the
equal fusion, on full reports, had been seen. The split rule is mechanical and is the one an earlier internal test used.

Added after the result was seen, on the advice of three separate automated reviews (nothing above was changed):
- What happened. Two settings tied exactly on the choosing half. The rule above picked the one with words counting
  more; on the held-out projects it was no better than the search as it was. The other tied setting was slightly
  better there. Neither difference can be told from chance, so the outcome is "inconclusive", not "the idea does not
  help", and the search was left as it was. Both are in the result ("tied_on_the_choosing_half").
- Precisions. "Alphabetical order" ignores case and sorts the full owner/name of the projects that have a replayed
  bug. "Full report" is the title and the start of the text (up to 1,600 characters). Settings are compared on exact
  counts of hits, not on rounded shares.
- For the next test of this kind: require the interval of the difference to exclude zero before changing anything,
  report every tied setting, and use projects that have not been looked at (all 22 now have been).

    uv run python tune_search.py collect    # the two raw rankings for every replayed bug (cached)
    uv run python tune_search.py verify     # is the offline fusion the search service's own?
    uv run python tune_search.py choose     # choose on one half, check on the other, write it down
"""
import datetime
import json
import random
import sys
import time
from pathlib import Path

import benchmark_memory as bm
from diastella_bench import search as recall

RANKINGS = bm.CACHE / "rankings"
WAYS = ("full report", "title alone")
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)          # what the ranking by words counts for; meaning counts 1
CONSTANTS = (2, 10, 60)                                         # the constant of reciprocal rank fusion (2 = the search service's own)
AS_IT_WAS = (1.0, 2)
DEEP = bm.DEPTH * 4                                             # how far each ranking is read, as in the product's search


def halves(repos: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(repos, key=str.lower)
    return ordered[0::2], ordered[1::2]


def question(bug: dict, way: str) -> str:
    return bug["text"] if way == "full report" else bug["text"].split("\n", 1)[0]


def _filter(key: str, t: int):
    from qdrant_client import models
    return models.Filter(must=[models.FieldCondition(key="repo", match=models.MatchValue(value=key)),
                               models.FieldCondition(key="day", range=models.Range(lt=t))])


def collect(repo: str) -> dict | None:
    """For every replayed bug of one project and each way of asking: the records closest by meaning and closest by
    words, among those dated before the bug, and what the search as it was returned (the service's own plain fusion,
    whose order among equal scores cannot be reproduced offline). Kept as numbers pointing into the list of record ids."""
    path = RANKINGS / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    data = bm.fetch(repo)
    if not data:
        return None
    from qdrant_client import models
    bugs, key, c = bm.prepare(data), bm.index(repo, data), recall.client()
    ids: list[str] = []
    reports: dict[str, list[str]] = {}
    seat: dict[str, int] = {}

    def keep(points) -> list[int]:
        out = []
        for p in points:
            rid = p.payload["id"]
            if rid not in seat:
                seat[rid] = len(ids); ids.append(rid); reports[rid] = p.payload["reports"]
            out.append(seat[rid])
        return out
    asked = {}
    for bid, bug, t, _, _ in bm.eligible(bugs):
        flt, asked[bid] = _filter(key, t), {}
        for way in WAYS:
            text = question(bug, way)
            meaning = c.query_points(bm.COLLECTION, query=recall._dense(text), using="dense", query_filter=flt, limit=DEEP, with_payload=["id", "reports"]).points
            words = c.query_points(bm.COLLECTION, query=recall._sparse(text), using="bm25", query_filter=flt, limit=DEEP, with_payload=["id", "reports"]).points
            served = c.query_points(bm.COLLECTION, limit=bm.DEPTH, with_payload=["id", "reports"], query=models.FusionQuery(fusion=models.Fusion.RRF),
                                    prefetch=[models.Prefetch(query=recall._sparse(text), using="bm25", filter=flt, limit=DEEP),
                                              models.Prefetch(query=recall._dense(text), using="dense", filter=flt, limit=DEEP)]).points
            asked[bid][way] = {"meaning": keep(meaning), "words": keep(words), "as it was": keep(served)}
    got = {"repo": repo, "ids": ids, "reports": reports, "asked": asked}
    RANKINGS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got))
    return got


def fuse(meaning: list, words: list, weight: float, constant: int, depth: int = bm.DEPTH) -> list:
    """Weighted reciprocal rank fusion as the search service computes it (found by comparing with its own output, see
    `verify`): a record scores, in each ranking it appears in, 1 / (constant - 1 + its rank / that ranking's weight),
    the first rank being 1, and the scores are added. A weight below 1 therefore pushes that ranking's records down
    the list (at 0.25, its first record counts like a fourth); at weight 1 this is the plain fusion."""
    total: dict = {}
    for ranking, w in ((words, weight), (meaning, 1.0)):
        for place, rid in enumerate(ranking if w else []):
            total[rid] = total.get(rid, 0.0) + 1.0 / (constant - 1 + (place + 1) / w)
    return sorted(total, key=lambda r: -total[r])[:depth]


def load(repos: list[str]) -> dict[str, list[dict]]:
    """Per project, one row per replayed bug: what its real fix changed, the earlier fixes, and the raw rankings."""
    out = {}
    for repo in repos:
        got, data = collect(repo), bm.fetch(repo)
        if not got or not got["asked"]:
            continue
        rows = []
        for bid, _, _, truth, earlier in bm.eligible(bm.prepare(data)):
            rows.append({"bug": bid, "truth": truth, "earlier": earlier, "asked": got["asked"][bid]})
        out[repo] = {"rows": rows, "ids": got["ids"], "reports": got["reports"]}
    return out


def hits(project: dict, way: str, weight: float, constant: int, measure: str = "file@5") -> list[bool]:
    ids, reports, out = project["ids"], project["reports"], []
    for row in project["rows"]:
        lists = row["asked"][way]
        ranked: list[str] = []
        for seat in (lists["as it was"] if (weight, constant) == AS_IT_WAS else fuse(lists["meaning"], lists["words"], weight, constant)):
            ranked += [r for r in reports[ids[seat]] if r not in ranked]
        out.append(bm.score(ranked, row["earlier"], row["truth"])[measure])
    return out


def share(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4)


def table(projects: dict, repos: list[str], settings: dict[str, tuple[float, int]]) -> dict:
    out = {}
    for way in WAYS:
        out[way] = {}
        for name, (weight, constant) in settings.items():
            out[way][name] = {m: share([h for r in repos for h in hits(projects[r], way, weight, constant, m)]) for m in ("file@3", "file@5", "file@10", "bug@1")}
    return out


def against(projects: dict, repos: list[str], a: tuple[float, int], b: tuple[float, int], rng: random.Random) -> dict:
    out = {}
    for way in WAYS:
        x = [h for r in repos for h in hits(projects[r], way, *a)]
        y = [h for r in repos for h in hits(projects[r], way, *b)]
        out[way] = {"difference": round((sum(x) - sum(y)) / len(x), 4), "hits": [sum(x), sum(y)], **bm.mcnemar(x, y),
                    "interval_95": bm.interval([float(v) for v in x], [float(v) for v in y], rng)}
    out["average of the two"] = {"difference": round(sum(out[way]["hits"][0] - out[way]["hits"][1] for way in WAYS) / (len(WAYS) * len(x)), 4)}
    return out


def passes(check: dict) -> bool:
    """The rule written before scoring: better on the average of the two ways of asking, and not more than half a
    point worse on either."""
    return bool(check["average of the two"]["difference"] > 0 and min(check[way]["difference"] for way in WAYS) >= -0.005)


def choose() -> dict:
    projects = load(bm.REPOS)
    settle, held = halves(list(projects))
    grid = [(w, k) for k in CONSTANTS for w in WEIGHTS if w or k == CONSTANTS[0]]       # with words at zero the constant changes nothing
    scored = []
    for weight, constant in grid:
        found = {way: [h for r in settle for h in hits(projects[r], way, weight, constant)] for way in WAYS}
        per_way = {way: share(found[way]) for way in WAYS}
        scored.append({"words_weight": weight, "constant": constant, **per_way, "hits": sum(sum(f) for f in found.values()),
                       "average": round(sum(sum(f) for f in found.values()) / sum(len(f) for f in found.values()), 4)})
    best = max(scored, key=lambda s: (s["hits"], s["words_weight"], -s["constant"]))
    tied = [(s["words_weight"], s["constant"]) for s in scored if s["hits"] == best["hits"]]
    chosen = (best["words_weight"], best["constant"])
    settings = {"search as it was (words and meaning equal)": AS_IT_WAS, "chosen weighting": chosen, "meaning alone": (0.0, 2)}
    rng = random.Random(20261117)
    held_table = table(projects, held, settings)
    check = against(projects, held, chosen, AS_IT_WAS, rng)
    gain, worst = check["average of the two"]["difference"], min(check[way]["difference"] for way in WAYS)
    others = [{"words_weight": w, "constant": k, **against(projects, held, (w, k), AS_IT_WAS, random.Random(20261117))} for w, k in tied if (w, k) != chosen]
    for other in others:
        other["would_have_passed_the_rule"] = passes(other)
    spans_zero = lambda c: all(c[way]["interval_95"][0] <= 0 <= c[way]["interval_95"][1] for way in WAYS)   # noqa: E731
    outcome = ("the search is changed" if passes(check) else
               "inconclusive: no tied setting can be told from the search as it was" if spans_zero(check) and all(spans_zero(o) for o in others) else
               "the chosen setting is not better")
    per_project = {r: {"bugs": len(projects[r]["rows"]),
                       **{way: {name: share(hits(projects[r], way, *s)) for name, s in settings.items()} for way in WAYS}} for r in held}
    return {"run_on": datetime.date.today().isoformat(), "rules_written_before_scoring": __doc__.split("\n\n")[2].replace("\n", " "),
            "grid": {"words_weight": WEIGHTS, "constant": CONSTANTS},
            "chosen_on": {"projects": settle, "bugs": sum(len(projects[r]["rows"]) for r in settle), "every_setting_file@5": scored},
            "chosen": {"words_weight": chosen[0], "constant": chosen[1]},
            "tied_on_the_choosing_half": [{"words_weight": w, "constant": k, "hits": best["hits"], "picked_by_the_rule": (w, k) == chosen} for w, k in tied],
            "on_the_half_used_to_choose_file@5": {way: {n: v["file@5"] for n, v in vals.items()} for way, vals in table(projects, settle, settings).items()},
            "held_out": {"projects": held, "bugs": sum(len(projects[r]["rows"]) for r in held), "results": held_table,
                         "chosen_against_search_as_it_was_file@5": check,
                         "other_tied_settings_against_search_as_it_was_file@5": others,
                         "chosen_against_meaning_alone_file@5": against(projects, held, chosen, (0.0, 2), rng),
                         "per_project_file@5": per_project},
            "decision": {"average_gain_held_out": gain, "worst_way_of_asking": worst, "change_the_search": passes(check), "outcome": outcome},
            "formula_check": verify(chosen),
            "notes": ["The search as it was is scored from the search service's own answers, recorded with the rankings, because the order "
                      "it gives records with exactly equal scores cannot be reproduced offline. Every other setting is computed offline "
                      "by a formula checked against the service (formula_check).",
                      "The decision rests on a handful of bugs: the difference on the held-out half is smaller than the movement of the "
                      "search as it was between two runs (order among equal scores). The test could not have told a gain of about one "
                      "point from none; its intervals are about one point either side.",
                      "Automated reviewers who scored the whole grid on the held-out half afterwards found about half the settings slightly above "
                      "the search as it was (0.6 to 1.2 points, almost all from report questions) and half below. That look is after "
                      "the fact and chooses nothing; it says the question is open for report questions and needs projects not yet seen."]}


def verify(chosen: tuple[float, int] | None = None, sample_every: int = 25) -> dict:
    """Ask the search service itself for the fused ranking and compare it with `fuse` on the same question: the order
    of the first records, and the scores (which do not depend on how equal scores are ordered)."""
    from qdrant_client import models
    c, tally, top = recall.client(), {}, set()
    pairs = [(0.0, 2), (0.05, 2), (0.3, 10), (0.5, 10), (0.5, 60), (0.75, 2), (1.0, 60)]
    pairs += [chosen] if chosen and chosen not in pairs and chosen != AS_IT_WAS else []
    tried = [(1.0, 2, models.FusionQuery(fusion=models.Fusion.RRF))] + [(w, k, models.RrfQuery(rrf=models.Rrf(k=k, weights=[w, 1.0]))) for w, k in pairs]
    for repo in bm.REPOS:
        got, data = collect(repo), bm.fetch(repo)
        if not got or not got["asked"]:
            continue
        key, bugs = "bench/" + repo, bm.prepare(data)
        for n, (bid, bug, t, _, _) in enumerate(bm.eligible(bugs)):
            if n % sample_every:
                continue
            way = WAYS[(n // sample_every) % len(WAYS)]                       # the two ways of asking in turn
            flt, text = _filter(key, t), question(bug, way)
            pre = [models.Prefetch(query=recall._sparse(text), using="bm25", filter=flt, limit=DEEP),
                   models.Prefetch(query=recall._dense(text), using="dense", filter=flt, limit=DEEP)]
            lists = got["asked"][bid][way]
            for weight, constant, query in tried:
                served = c.query_points(bm.COLLECTION, limit=bm.DEPTH, with_payload=["id"], query=query, prefetch=pre).points
                ours = [got["ids"][s] for s in fuse(lists["meaning"], lists["words"], weight, constant)]
                theirs = [p.payload["id"] for p in served]
                expect: dict = {}
                for ranking, w in ((lists["words"], weight), (lists["meaning"], 1.0)):
                    for place, seat in enumerate(ranking if w else []):
                        expect[got["ids"][seat]] = expect.get(got["ids"][seat], 0.0) + 1.0 / (constant - 1 + (place + 1) / w)
                gap = max((abs(p.score - expect.get(p.payload["id"], 0.0)) for p in served[:20]), default=0.0)
                name = "plain fusion" if isinstance(query, models.FusionQuery) else f"words {weight}, constant {constant}"
                one = tally.setdefault(name, {"questions": 0, "same_first_five": 0, "same_first_twenty": 0, "largest_score_gap": 0.0})
                one["questions"] += 1; one["same_first_five"] += ours[:5] == theirs[:5]; one["same_first_twenty"] += ours[:20] == theirs[:20]
                one["largest_score_gap"] = round(max(one["largest_score_gap"], gap), 7)
                if isinstance(query, models.FusionQuery) and served:
                    top.add(round(served[0].score, 4))
    return {"offline_fusion_against_the_search_service": tally, "highest_scores_of_the_plain_fusion": sorted(top)[-3:],
            "note": "Where the order differs and the scores do not, the records have exactly equal scores and the service orders them as it likes."}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "choose"
    recall.COLLECTION = bm.COLLECTION
    if what == "collect":
        for repo in sys.argv[2:] or bm.REPOS:
            started = time.time()
            got = collect(repo)
            print(f"  {repo}: {len(got['asked']) if got else 0} bugs asked two ways ({time.time() - started:.0f}s)", flush=True)
    elif what == "verify":
        print(json.dumps(verify((0.5, 10)), indent=1))
    else:
        result = choose()
        bm.OUT.mkdir(parents=True, exist_ok=True)
        (bm.OUT / f"memory-weighting-{result['run_on']}.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k not in ("rules_written_before_scoring",)}, indent=1))
