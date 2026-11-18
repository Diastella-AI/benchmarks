"""The search over a project's records, as the hosted service runs it, in Qdrant.

One collection holds every project, with the repository as the tenant key. Each record (a bug report, or a merged
change that fixed one) is stored with two vectors: `dense`, from the embedding model all-MiniLM-L6-v2, and `bm25`, the
keyword vector Qdrant computes itself. A search asks both and fuses the two rankings by reciprocal rank fusion in one
query. This file is the part of Diastella's search layer that the benchmarks call, cut out unchanged in what it does."""
from __future__ import annotations
import os
import threading


COLLECTION = os.getenv("QDRANT_COLLECTION", "diastella_records")


DENSE_MODEL, DENSE_LOCAL, DENSE_DIM = "sentence-transformers/all-minilm-l6-v2", "sentence-transformers/all-MiniLM-L6-v2", 384

BM25_MODEL = "qdrant/bm25"


_client = None


_lock = threading.Lock()


_ready: set[str] = set()


_calls = threading.RLock()


def _serial(fn):
    """One Qdrant call at a time: a local embedding model must not be loaded or used by two threads at once."""
    import functools

    @functools.wraps(fn)
    def run(*args, **kwargs):
        with _calls:
            return fn(*args, **kwargs)
    return run


def _cloud() -> bool:
    """Qdrant Cloud makes the dense vectors itself; anywhere else this machine does, if it can."""
    setting = os.getenv("QDRANT_INFERENCE", "auto")
    return setting == "cloud" or (setting == "auto" and "cloud.qdrant.io" in os.getenv("QDRANT_URL", ""))


def client():
    global _client
    with _lock:
        if _client is None:
            from qdrant_client import QdrantClient
            _client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.getenv("QDRANT_API_KEY") or None,
                                   cloud_inference=_cloud(), timeout=30)
        return _client


def _dense_model() -> str | None:
    if _cloud():
        return DENSE_MODEL
    try:
        import fastembed  # noqa: F401
        return DENSE_LOCAL
    except ImportError:
        return None


def _dense(text: str):
    from qdrant_client import models
    model = _dense_model()
    return models.Document(text=text[:2000], model=model) if model else None


def _sparse(text: str):
    from qdrant_client import models
    return models.Document(text=text[:4000], model=BM25_MODEL)


def ensure() -> None:
    """Create the collection on first use (again if the dense model changed), with an index per filtered field."""
    if COLLECTION in _ready:
        return
    from qdrant_client import models
    c = client()
    if c.collection_exists(COLLECTION) and c.get_collection(COLLECTION).config.params.vectors["dense"].size != DENSE_DIM:
        c.delete_collection(COLLECTION)
    if not c.collection_exists(COLLECTION):
        c.create_collection(COLLECTION, vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
                            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        # one collection for every project: `repo` is the tenant key, so each project's records are stored together
        c.create_payload_index(COLLECTION, "repo", models.KeywordIndexParams(type="keyword", is_tenant=True))
        for field in ("kind", "systems", "stamp"):
            c.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    for field in ("private", "library"):                      # also on a collection made before these existed
        try:
            c.create_payload_index(COLLECTION, field, models.PayloadSchemaType.BOOL)
        except Exception:
            pass
    _ready.add(COLLECTION)


def _stamp(snapshot: dict) -> str:
    return f"{snapshot.get('fetched_at')}/v{snapshot.get('connector_version', 1)}"


def key(snapshot: dict) -> str:
    """What a project's records are stored under: the repository's name, unless the snapshot names another key."""
    return snapshot.get("recall_key") or snapshot["repo"]


def _records(snapshot: dict):
    """(id, text, payload) for every bug report and every fix in a repository snapshot."""
    repo, stamp, fixes = key(snapshot), _stamp(snapshot), {}
    private, library = bool(snapshot.get("private")), bool(snapshot.get("library"))
    for i in snapshot["incidents"]:
        rid = f"ISSUE-{i['number']}"
        # in the hosted service a library's reports carry their fixes with them; the benchmarks never set `library`
        extra = {"library": True, "fixes": [{"id": f"PR-{f['number']}", "url": f["url"], "title": f["title"], "merged_at": f["merged_at"],
                                             "author": f.get("author"), "files": (f.get("files") or [])[:8]} for f in i["fixes"]]} if library else {}
        yield rid, f"{i['title']}\n{i.get('text') or ''}", {
            "repo": repo, "private": private, "stamp": stamp, "kind": "report", "id": rid, "title": i["title"], "url": i["url"], "status": i["status"],
            "opened_at": i["opened_at"], "systems": i["services"], "reports": [rid], **extra}
        for f in i["fixes"]:
            fixes.setdefault(f["number"], (f, []))[1].append(rid)
    for number, (f, reports) in fixes.items():
        fid = f"PR-{number}"
        yield fid, f"{f['title']}\n{f.get('text') or ''}\n{' '.join(f.get('files') or [])}", {
            "repo": repo, "private": private, "stamp": stamp, "kind": "fix", "id": fid, "title": f["title"], "url": f["url"], "status": "merged",
            "opened_at": f["merged_at"], "systems": f["services"], "reports": reports, **({"library": True} if library else {})}


def _repo_filter(repo: str, elsewhere: bool = False, kind: str | None = None):
    from qdrant_client import models
    same = models.FieldCondition(key="repo", match=models.MatchValue(value=repo))
    must = [] if elsewhere else [same]
    if kind:
        must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
    # "Elsewhere" means other PUBLIC projects: a private repository's records are never offered to another project.
    hidden = models.FieldCondition(key="private", match=models.MatchValue(value=True))
    # nor a library's records, which the hosted service searches on their own
    library = models.FieldCondition(key="library", match=models.MatchValue(value=True))
    return models.Filter(must=must, must_not=[same, hidden, library] if elsewhere else [])


@_serial
def search(repo: str, query: str, limit: int = 8, kind: str | None = None, elsewhere: bool = False) -> list[dict]:
    """The records closest to `query` by meaning and by words, in this project (or, with `elsewhere`, in every other
    one). Each result is the stored payload plus a rank score."""
    from qdrant_client import models
    ensure()
    flt, n, dense = _repo_filter(repo, elsewhere, kind), max(1, min(limit, 30)), _dense(query)
    found = client().query_points(
        COLLECTION, limit=n, with_payload=True, query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=[models.Prefetch(query=_sparse(query), using="bm25", filter=flt, limit=n * 4),
                  *([models.Prefetch(query=dense, using="dense", filter=flt, limit=n * 4)] if dense else [])])
    return [{**p.payload, "score": round(p.score, 4)} for p in found.points]
