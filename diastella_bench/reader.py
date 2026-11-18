"""Reading a public repository's bug reports and their fixes from GitHub.

This is the part of Diastella's connector that the benchmarks call, cut out of the hosted service unchanged in what it
does: which label a project uses for bugs, its bug-labelled reports with the merged changes that closed them (title,
start of the text, files changed), and the parts of the code a changed file belongs to. Only public data is read, with
GITHUB_TOKEN or the logged-in `gh` command."""
from __future__ import annotations
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import http.client
import urllib.request



DATASTORE = re.compile(r"(db|sql|valkey|redis|kafka|opensearch|mongo)", re.I)


ISSUES_QUERY = """
query($owner: String!, $name: String!, $label: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 40, after: $cursor, labels: [$label], orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url state createdAt closedAt bodyText
        author { login }
        closedByPullRequestsReferences(first: 5) {
          nodes { number title url mergedAt bodyText changedFiles author { login } files(first: 100) { nodes { path } } }
        }
      }
    }
  }
}"""


_TOKEN: str | None = None


_TOKEN_LOCK = threading.Lock()


class GitHubError(RuntimeError):
    pass



def _token() -> str:
    """A GitHub token with public read access: GITHUB_TOKEN on a server, or the logged-in `gh` CLI on a laptop."""
    global _TOKEN
    with _TOKEN_LOCK:                                         # two requests at once must not see a half-set value
        if _TOKEN is None:
            token = (os.getenv("GITHUB_TOKEN") or "").strip()
            if not token:
                try:
                    token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10).stdout.strip()
                except Exception:
                    token = ""
            _TOKEN = token
        if _TOKEN and not (_TOKEN.isascii() and _TOKEN.isprintable() and " " not in _TOKEN):
            # e.g. the dots of a hidden value copied from a settings page in place of the value itself
            raise GitHubError("GITHUB_TOKEN is not a real token (it contains characters a token cannot have).")
        return _TOKEN


def _api(path: str, accept: str = "application/vnd.github+json", body: dict | None = None, attempts: int = 4):
    """One GitHub API call. Returns parsed JSON (or text for raw content), or None for 404. Blips and rate-limit pauses are retried."""
    headers = {"Accept": accept, "User-Agent": "diastella-benchmarks", "X-GitHub-Api-Version": "2022-11-28"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"https://api.github.com/{path}", headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode()
                return text if "raw" in accept else json.loads(text)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (401, 403) and i == attempts - 1:
                raise GitHubError("GitHub refused the request (rate limit reached or no access).") from e
            if e.code == 409:                                  # an empty repository
                return None
            if i == attempts - 1:
                raise GitHubError(f"GitHub answered {e.code} for {path.split('?')[0]}") from e
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError) as e:   # incl. a reply cut off midway
            if i == attempts - 1:
                raise GitHubError("Could not reach GitHub.") from e
        time.sleep(2 * (i + 1) ** 2)


def _raw(repo: str, path: str) -> str | None:
    return _api(f"repos/{repo}/contents/{path}", accept="application/vnd.github.raw")


CONNECTOR_VERSION = 5          # bump when the map a repository produces changes, so stored maps are rebuilt


COMPOSE_FILES = ("compose.yaml", "compose.yml", "docker-compose.yml", "docker-compose.yaml", "compose.full.yaml")


SYSTEM_ROOTS = ("services", "apps", "packages", "cmd", "src", "modules", "components", "plugins", "crates", "libs")


# Folders that never hold a system of their own, at any depth.
NEVER = re.compile(r"^(\..*|__.*__|docs?|doc[_-].*|docs[_-].*|tests?|testing|test[_-].*|.*[_-]tests?|e2e|examples?|samples?|demos?|bench|benchmarks?|"
                   r"changelog.*|scripts?|vendor|node_modules|dist|build|third[_-]party|assets|static|public|images?|img|fixtures|"
                   r"migrations|locales|i18n|licenses?|site|website|proposals|design)$", re.I)


# Top-level folders that support the project without being a system (fine as a part deeper down, e.g. pkg/utils).
NOT_TOP = {"hack", "tools", "ci", "bin", "internal", "pb", "proto", "shared", "common", "utils", "util", "types", "config", "configs",
           "docker", "deploy", "deployment", "deployments", "charts", "helm", "k8s", "kubernetes", "manifests", "templates",
           "integration", "spec", "specs", "release", "releases"}


CODE = {"py", "pyi", "js", "jsx", "ts", "tsx", "mjs", "go", "rs", "java", "kt", "kts", "scala", "rb", "php", "cs", "fs", "c", "cc", "cpp",
        "cxx", "h", "hpp", "m", "mm", "swift", "ex", "exs", "erl", "clj", "cljs", "hs", "ml", "lua", "pl", "r", "jl", "dart", "sol",
        "vue", "svelte", "sql", "sh", "zig", "nim", "groovy"}


MAX_SYSTEMS = 40


def _code_paths(repo: str, branch: str) -> tuple[list[str], list[str]] | None:
    """Every source file and every folder in the repository (one API call), or None when it is too large to list at once."""
    tree = _api(f"repos/{repo}/git/trees/{urllib.parse.quote(branch)}?recursive=1")
    if not tree or tree.get("truncated"):
        return None
    code, folders = [], []
    for t in tree["tree"]:
        parts = t["path"].split("/")
        if t["type"] == "tree" and not any(NEVER.match(x) for x in parts):
            folders.append(t["path"])
        elif t["type"] == "blob" and "." in parts[-1] and parts[-1].rsplit(".", 1)[-1].lower() in CODE and not any(NEVER.match(x) for x in parts[:-1]):
            code.append(t["path"])
    return code, folders


def folder_systems(paths: list[str]) -> list[dict]:
    """Choose the folders to treat as systems, from the list of source files.

    Start from the top-level code folders (opening the usual roots such as services/ or packages/). While that gives
    fewer than four systems, open the largest one: a single-package library becomes its sub-packages. A parent that
    also has files of its own stays as a system for the whole package."""
    paths = [p for p in paths if not any(NEVER.match(x) for x in p.split("/")[:-1])]

    def kids(path: str, parts_only: bool = False) -> dict[str, int]:
        pre = path + "/" if path else ""
        seen: dict[str, int] = {}
        for p in paths:
            if p.startswith(pre) and "/" in p[len(pre):]:
                name = p[len(pre):].split("/", 1)[0]
                seen[pre + name] = seen.get(pre + name, 0) + 1
        if parts_only:                                        # the parts of a package: look through src/ and lib/, skip private ones
            for holder in [k for k in seen if k.rsplit("/", 1)[-1] in ("src", "lib", "source", "pkg")]:
                inner = kids(holder, True)
                if len(inner) >= 2:
                    del seen[holder]; seen.update(inner)
            seen = {k: n for k, n in seen.items() if not k.rsplit("/", 1)[-1].startswith("_")}
        return seen

    def direct(path: str) -> int:
        return sum(1 for p in paths if p.startswith(path + "/") and "/" not in p[len(path) + 1:])

    top = {p: n for p, n in kids("").items() if p.lower() not in NOT_TOP}
    systems: dict[str, int] = {}
    for p, n in top.items():
        inner = kids(p) if p in SYSTEM_ROOTS else {}
        systems.update(inner if len(inner) >= 2 else {p: n})
    if not systems:
        systems = dict(kids(""))
    closed: set[str] = set()                                  # parents kept for their own files: not opened again
    for _ in range(4):
        if len(systems) >= 3:
            break
        openable = [(n, p) for p, n in systems.items() if p not in closed and kids(p, True)]
        if not openable:
            break
        _, parent = max(openable)
        inner = kids(parent, True)
        del systems[parent]
        systems.update(inner)
        if len(inner) >= 2 and direct(parent) >= 3:
            systems[parent] = sum(1 for p in paths if p.startswith(parent + "/")); closed.add(parent)
    modules: set[str] = set()
    if len(systems) < 3:
        # A single package with its code in files, not folders (most small libraries and applications): each source file
        # of the folder that holds the most of them is a system of its own, next to any sub-folders found above.
        def own_files(folder: str) -> list[str]:
            pre = folder + "/" if folder else ""
            return [p for p in paths if p.startswith(pre) and "/" not in p[len(pre):] and _is_module(p)]
        folders = {p.rpartition("/")[0] for p in paths}
        folders = [f for f in folders if f.split("/")[0].lower() not in NOT_TOP]
        home = max(folders, key=lambda f: (len(own_files(f)), -len(f)), default=None)
        if home is not None and len(own_files(home)) >= 3:
            # the package itself and anything above it give way to its files
            systems = {p: n for p, n in systems.items() if p != home and not (home + "/").startswith(p + "/")}
            modules = set(own_files(home))
            systems.update({m: 1 for m in modules})
    ranked = sorted(systems.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_SYSTEMS]
    stem = lambda p: p.rsplit("/", 1)[-1].rsplit(".", 1)[0] if p in modules else p.rsplit("/", 1)[-1]      # noqa: E731
    names: dict[str, int] = {}
    for p, _ in ranked:
        names[stem(p)] = names.get(stem(p), 0) + 1
    out = []
    for p, n in ranked:
        base = stem(p)
        ident = base if names[base] == 1 else "-".join(p.split("/")[-2:])
        out.append({"id": re.sub(r"[^\w.-]", "-", ident), "name": ident, "path": p, "files": n,
                    "kind": "module" if p in modules else "datastore" if DATASTORE.search(base) else "package" if p in closed else "service"})
    return out


NOT_A_MODULE = re.compile(r"^(_.*|test_.*|.*_test|.*\.(test|spec|d|min|config)|setup|conftest|noxfile|fabfile|manage|gulpfile|gruntfile|webpack.*|vite.*|rollup.*|babel.*|jest.*|eslint.*|tailwind.*|postcss.*|version|constants?)$", re.I)


def _is_module(path: str) -> bool:
    """A source file that is a part of the program in its own right: not a test, a build script or a private helper."""
    name = path.rsplit("/", 1)[-1]
    stem, _, ext = name.rpartition(".")
    return bool(stem) and ext.lower() in CODE and ext.lower() not in ("sh", "sql", "pyi", "h", "hpp") and not NOT_A_MODULE.match(stem)


def _dirs(repo: str, path: str = "") -> list[str]:
    listing = _api(f"repos/{repo}/contents/{path}".rstrip("/")) or []
    return [d["name"] for d in listing if isinstance(d, dict) and d.get("type") == "dir" and not d["name"].startswith(".")]


def _systems(repo: str, branch: str) -> tuple[list[dict], list[dict], str]:
    """Find the repository's systems and how they depend on each other.

    If the repository has compose files, its services are the systems and `depends_on` / `*_ADDR` settings are the
    dependencies. Otherwise the systems are folders chosen by `folder_systems`, and no dependencies are claimed."""
    import yaml

    listed = _code_paths(repo, branch)
    if listed is None:                                        # too large to list: fall back to the top-level folders
        top = [d for d in _dirs(repo) if not NEVER.match(d) and d.lower() not in NOT_TOP]
        listed = ([f"{d}/x.py" for d in top], top)
    paths, folders = listed
    by_folder: dict[str, str] = {}                            # folder name -> its shallowest path, for matching services
    for path in folders:
        if path.count("/") <= 2:
            name = path.rsplit("/", 1)[-1]
            if name not in by_folder or path.count("/") < by_folder[name].count("/"):
                by_folder[name] = path

    defs: dict[str, dict] = {}
    for f in COMPOSE_FILES:
        text = _raw(repo, f)
        try:
            parsed = (yaml.safe_load(text) or {}).get("services") or {} if text else {}
        except Exception:
            parsed = {}
        for name, d in parsed.items():
            if not isinstance(d, dict):
                continue
            cur = defs.setdefault(name, {"depends_on": [], "environment": [], "text": ""})
            cur["depends_on"] += list(d.get("depends_on") or [])
            env = d.get("environment") or []
            cur["environment"] += env if isinstance(env, list) else [f"{k}={v}" for k, v in env.items()]
            cur["text"] += json.dumps(d, default=str)

    if len(defs) < 2:                                         # no usable compose file: folders are the systems
        return [{k: v for k, v in s.items() if k != "files"} for s in folder_systems(paths)], [], "folders"

    declared = {name: {dep for dep in d["depends_on"] if dep in defs and dep != name} for name, d in defs.items()}

    def reaches(a: str, b: str, seen: set | None = None) -> bool:
        seen = seen if seen is not None else set()
        if a == b:
            return True
        seen.add(a)
        return any(n not in seen and reaches(n, b, seen) for n in declared.get(a, ()))

    systems, depends = [], []
    for name, d in list(defs.items())[:MAX_SYSTEMS]:
        words = set(re.findall(r"[\w-]+", d["text"]))               # a folder named in the service's build, image or volumes
        mentioned = [f for f in by_folder if len(f) > 3 and f in words]
        folder = by_folder.get(name) or (by_folder[mentioned[0]] if mentioned else None)
        kind = "datastore" if DATASTORE.search(name) else "service" if folder else "infrastructure"
        systems.append({"id": name, "name": name, "path": folder, "kind": kind})
        seen = set()
        for dep in d["depends_on"]:
            if dep in defs and dep != name and dep not in seen:
                seen.add(dep); depends.append({"from": name, "to": dep, "basis": "compose depends_on"})
        for e in d["environment"]:
            key = str(e).split("=")[0]
            target = key[:-5].lower().replace("_", "-") if key.endswith("_ADDR") else None
            # a setting that points back at something which already depends on this system is a reference, not a dependency
            if target in defs and target != name and target not in seen and not reaches(target, name):
                seen.add(target); declared[name].add(target)
                depends.append({"from": name, "to": target, "basis": f"setting {key}"})
    ids = {s["id"] for s in systems}
    return systems, [d for d in depends if d["from"] in ids and d["to"] in ids], "compose"


def _bug_label(repo: str) -> str | None:
    """The label this project uses for bugs: the most used label whose name contains 'bug'."""
    owner, name = repo.split("/")
    q = "query($o:String!,$n:String!){repository(owner:$o,name:$n){labels(first:30,query:\"bug\"){nodes{name issues{totalCount}}}}}"
    data = _api("graphql", body={"query": q, "variables": {"o": owner, "n": name}})
    labels = ((((data or {}).get("data") or {}).get("repository") or {}).get("labels") or {}).get("nodes") or []
    labels = [l for l in labels if l["issues"]["totalCount"] > 0]
    return max(labels, key=lambda l: l["issues"]["totalCount"])["name"] if labels else None


def _service_of_paths(paths: list[str], by_path: dict[str, str]) -> list[str]:
    """The systems a set of changed files belongs to: each file counts for the most specific folder that holds it."""
    hit = set()
    for p in paths:
        best = max((prefix for prefix in by_path if p == prefix or p.startswith(prefix + "/")), key=len, default=None)
        if best:
            hit.add(by_path[best])
    return sorted(hit)


COMMON_WORDS = {"types", "utils", "util", "helpers", "common", "config", "errors", "exceptions", "requests", "responses", "models", "tests",
                "index", "main", "server", "client", "core", "base", "state", "store", "views", "forms", "tasks", "events", "status", "logging"}


def _service_in_title(title: str, services: list[dict]) -> list[str]:
    t = title.lower()
    found = []
    for s in sorted(services, key=lambda s: -len(s["id"])):
        if s.get("kind") == "module" and (len(s["id"]) < 5 or s["id"].lower() in COMMON_WORDS):
            continue                                          # a file called "types" or "utils" is not what a title means by the word
        forms = {s["id"], s["id"].replace("-", " "), s["id"].replace("-", "")}
        if any(re.search(rf"(?<![a-z]){re.escape(f)}(?![a-z])", t) for f in forms):
            found.append(s["id"])
    return found[:2]


def _excerpt(text: str | None, limit: int) -> str | None:
    """The start of a report's or a fix's own words, on one line, cut at a word."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    return text or None


def _incidents(repo: str, label: str, services: list[dict], max_pages: int = 12) -> list[dict]:
    owner, name = repo.split("/")
    by_path = {p: s["id"] for s in services if s["path"] for p in [s["path"], *s.get("earlier_paths", [])]}
    out, cursor = [], None
    for _ in range(max_pages):
        data = _api("graphql", body={"query": ISSUES_QUERY, "variables": {"owner": owner, "name": name, "label": label, "cursor": cursor}})
        if not data or data.get("errors"):
            raise GitHubError("GitHub could not list this repository's issues.")
        page = data["data"]["repository"]["issues"]
        for i in page["nodes"]:
            fixes = []
            for pr in i["closedByPullRequestsReferences"]["nodes"]:
                if not pr or not pr.get("mergedAt"):
                    continue
                paths = [f["path"] for f in pr["files"]["nodes"]]
                fixes.append({"number": pr["number"], "title": pr["title"], "url": pr["url"], "merged_at": pr["mergedAt"][:10],
                              "author": (pr.get("author") or {}).get("login"),
                              "text": _excerpt(pr.get("bodyText"), 1200), "files": paths[:12], "changed_files": pr.get("changedFiles"),
                              "services": _service_of_paths(paths, by_path)})
            from_fix = sorted({s for f in fixes for s in f["services"]})
            affected = from_fix or _service_in_title(i["title"], services)
            out.append({"number": i["number"], "title": i["title"], "url": i["url"], "status": i["state"].lower(),
                        "opened_at": i["createdAt"][:10], "closed_at": (i["closedAt"] or "")[:10] or None,
                        "reporter": (i.get("author") or {}).get("login"), "text": _excerpt(i.get("bodyText"), 1600), "services": affected,
                        "basis": "files changed by the fix" if from_fix else "named in the title" if affected else None,
                        "fixes": fixes})
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return out
