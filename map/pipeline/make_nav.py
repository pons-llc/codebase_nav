#!/usr/bin/env python3
"""make_nav.py — Generate a compact, token-efficient CODEBASE_NAV.md for any repo.

Single-file, dependency-optional. The pipeline degrades gracefully:

  Embedding : fastembed (BAAI/bge-small-en-v1.5)  ->  sklearn TF-IDF  ->  none
  Clustering: UMAP + HDBSCAN  ->  sklearn KMeans  ->  import-graph community (pure)
  Ranking   : networkx PageRank  ->  pure-Python PageRank

Even with zero third-party packages installed it produces a useful nav file
using only the Python standard library.

Usage:
    python make_nav.py --repo /path/to/repo [--out DIR] [--detail std]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Optional dependencies — detected at runtime, never required.
# ---------------------------------------------------------------------------
try:
    import numpy as np  # noqa
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

try:
    import networkx as nx  # noqa
    HAVE_NX = True
except Exception:
    HAVE_NX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
    from sklearn.cluster import KMeans  # noqa
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    import umap  # noqa
    import hdbscan  # noqa
    HAVE_UMAP_HDBSCAN = True
except Exception:
    HAVE_UMAP_HDBSCAN = False

# fastembed is heavy; only probe lazily inside embed().


# ---------------------------------------------------------------------------
# Language definitions
# ---------------------------------------------------------------------------
LANG_BY_EXT = {
    ".py": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
}

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "target", "vendor", ".idea",
    ".vscode", "coverage", ".mypy_cache", ".pytest_cache", ".tox", "out",
    "bin", "obj", ".cache", "site-packages", "migrations",
}

CODE_EXTS = set(LANG_BY_EXT)

# Keyword buckets used to name clusters heuristically.
KEYWORD_BUCKETS = [
    ("Authentication", ("auth", "login", "session", "token", "jwt", "oauth", "password", "credential")),
    ("API/Server", ("api", "server", "route", "router", "endpoint", "controller", "handler", "rest", "graphql")),
    ("Database/Models", ("model", "schema", "db", "database", "orm", "entity", "repository", "migration", "query")),
    ("UI Components", ("component", "view", "page", "widget", "ui", "render", "screen", "layout")),
    ("State Management", ("store", "state", "reducer", "context", "redux", "vuex", "signal", "atom")),
    ("Search/RAG", ("search", "index", "embed", "vector", "rag", "retriev", "rank")),
    ("Chat/LLM", ("chat", "llm", "prompt", "completion", "message", "agent", "model")),
    ("Config/Settings", ("config", "setting", "env", "constant", "option", "preference")),
    ("Utilities/Helpers", ("util", "helper", "common", "shared", "lib", "tool", "format")),
    ("Testing", ("test", "spec", "mock", "fixture", "stub")),
    ("Networking", ("http", "client", "request", "fetch", "socket", "ws", "websocket", "rpc")),
    ("Storage/Files", ("file", "storage", "upload", "download", "blob", "fs", "disk", "cache")),
    ("Tasks/Jobs", ("task", "job", "queue", "worker", "cron", "schedule", "celery")),
    ("Validation", ("valid", "verify", "check", "guard", "assert", "sanitize")),
]

MAX_FILE_BYTES = 400_000  # skip giant / generated files
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_files(repo, max_files):
    out = []
    repo = os.path.abspath(repo)
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in CODE_EXTS:
                continue
            if fn.endswith((".min.js", ".min.ts", ".d.ts", ".bundle.js")):
                continue
            full = os.path.join(root, fn)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            out.append(os.path.relpath(full, repo))
    out.sort()
    if max_files and len(out) > max_files:
        # keep a representative spread, but prefer non-test source first
        out = sorted(out, key=lambda p: ("test" in p.lower(), p))[:max_files]
        out.sort()
    return repo, out


# ---------------------------------------------------------------------------
# Per-file parsing
# ---------------------------------------------------------------------------
PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+(\.*[\w.]+)\s+import|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))", re.M)
PY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)", re.M)
PY_FUNC_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", re.M)

JS_IMPORT_RE = re.compile(r"""(?:import\b[^'"]*?from\s*['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)|import\(\s*['"]([^'"]+)['"]\s*\)|export\b[^'"]*?from\s*['"]([^'"]+)['"])""")
JS_CLASS_RE = re.compile(r"\b(?:export\s+(?:default\s+)?)?class\s+(\w+)")
JS_FUNC_RE = re.compile(r"\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)")
JS_ARROW_RE = re.compile(r"\b(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>")
JS_EXPORT_RE = re.compile(r"\bexport\s+(?:const|let|var|function|class|default|async\s+function)?\s*(\w+)?")

GENERIC_IMPORT_RE = {
    "go": re.compile(r'^\s*(?:import\s+)?"([^"]+)"', re.M),
    "java": re.compile(r"^\s*import\s+([\w.]+)", re.M),
    "ruby": re.compile(r"^\s*require(?:_relative)?\s+['\"]([^'\"]+)['\"]", re.M),
    "rust": re.compile(r"^\s*use\s+([\w:]+)", re.M),
}
GENERIC_CLASS_RE = {
    "go": re.compile(r"\btype\s+(\w+)\s+struct"),
    "java": re.compile(r"\b(?:public\s+|abstract\s+|final\s+)*class\s+(\w+)"),
    "ruby": re.compile(r"^\s*class\s+(\w+)", re.M),
    "rust": re.compile(r"\b(?:pub\s+)?struct\s+(\w+)"),
}
GENERIC_FUNC_RE = {
    "go": re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?(\w+)\s*\(([^)]*)\)"),
    "java": re.compile(r"\b(?:public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(([^)]*)\)"),
    "ruby": re.compile(r"^\s*def\s+(\w+)", re.M),
    "rust": re.compile(r"\b(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)"),
}


def first_summary(lang, text):
    """Extract a one-line human summary from docstring / leading comment."""
    lines = text.splitlines()
    if lang == "python":
        m = re.search(r'^\s*(?:[rRbBuU]{0,2})("""|\'\'\')(.*?)\1', text, re.S)
        if m:
            doc = m.group(2).strip().splitlines()
            if doc and doc[0].strip():
                return doc[0].strip()
    # block / line comment near the top
    for ln in lines[:15]:
        s = ln.strip()
        if s.startswith(("/**", "/*", "*", "//", "#", '"""', "///")):
            s = re.sub(r"^[/*#\"\s]+", "", s).strip()
            if len(s) > 4 and not s.startswith(("eslint", "@ts", "type:", "!")):
                return s
    return ""


def parse_file(repo, rel, lang):
    full = os.path.join(repo, rel)
    try:
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    loc = text.count("\n") + 1
    imports, classes, funcs = [], [], []

    if lang == "python":
        for m in PY_IMPORT_RE.finditer(text):
            mod = m.group(1) or m.group(2) or ""
            for part in mod.split(","):
                part = part.strip()
                if part:
                    imports.append(part)
        classes = PY_CLASS_RE.findall(text)
        for ind, name, args in PY_FUNC_RE.findall(text):
            if len(ind) == 0:  # top-level only for signatures
                sig = re.sub(r"\s+", " ", args).strip()
                funcs.append((name, sig))
    elif lang in ("js", "ts"):
        for m in JS_IMPORT_RE.finditer(text):
            spec = m.group(1) or m.group(2) or m.group(3) or m.group(4)
            if spec:
                imports.append(spec)
        classes = JS_CLASS_RE.findall(text)
        for name, args in JS_FUNC_RE.findall(text):
            funcs.append((name, re.sub(r"\s+", " ", args).strip()))
        for name, args in JS_ARROW_RE.findall(text):
            funcs.append((name, re.sub(r"\s+", " ", args).strip()))
    else:
        imp_re = GENERIC_IMPORT_RE.get(lang)
        if imp_re:
            imports = imp_re.findall(text)
        cls_re = GENERIC_CLASS_RE.get(lang)
        if cls_re:
            classes = cls_re.findall(text)
        fn_re = GENERIC_FUNC_RE.get(lang)
        if fn_re:
            for found in fn_re.findall(text):
                if isinstance(found, tuple):
                    funcs.append((found[0], re.sub(r"\s+", " ", found[1]).strip()))
                else:
                    funcs.append((found, ""))

    # de-dup, cap
    seen = set()
    uclasses = [c for c in classes if not (c in seen or seen.add(c))]
    seen = set()
    ufuncs = [f for f in funcs if not (f[0] in seen or seen.add(f[0]))]

    return {
        "rel": rel,
        "lang": lang,
        "loc": loc,
        "imports": imports,
        "classes": uclasses,
        "funcs": ufuncs,
        "summary": first_summary(lang, text),
    }


# ---------------------------------------------------------------------------
# Import resolution -> directed graph (importer -> imported)
# ---------------------------------------------------------------------------
def build_graph(repo, infos):
    rel_set = {i["rel"] for i in infos}

    # Python module-path index: "pkg/sub/mod" and "pkg/sub" (for __init__)
    py_index = {}
    for rel in rel_set:
        if rel.endswith(".py"):
            noext = rel[:-3]
            key = noext.replace(os.sep, "/")
            if key.endswith("/__init__"):
                key = key[: -len("/__init__")]
            py_index[key] = rel
            py_index[key.replace("/", ".")] = rel

    # JS index: path without extension and index files
    js_index = {}
    for rel in rel_set:
        base, ext = os.path.splitext(rel)
        if ext.lower().lstrip(".") in ("js", "jsx", "ts", "tsx", "mjs", "cjs"):
            js_index[base.replace(os.sep, "/")] = rel
            b = base.replace(os.sep, "/")
            if b.endswith("/index"):
                js_index[b[: -len("/index")]] = rel

    edges = Counter()

    def resolve_relative(curdir, spec):
        target = os.path.normpath(os.path.join(curdir, spec)).replace(os.sep, "/")
        if target in js_index:
            return js_index[target]
        # try as directory index
        if target + "/index" in [k for k in js_index]:
            return js_index.get(target + "/index")
        if (target + "/index") in js_index:
            return js_index[target + "/index"]
        return None

    for info in infos:
        rel = info["rel"]
        curdir = os.path.dirname(rel)
        for spec in info["imports"]:
            tgt = None
            if info["lang"] == "python":
                if spec.startswith("."):
                    # relative import
                    up = len(spec) - len(spec.lstrip("."))
                    base = curdir
                    for _ in range(up - 1):
                        base = os.path.dirname(base)
                    rest = spec.lstrip(".").replace(".", "/")
                    cand = (base + "/" + rest).strip("/") if rest else base
                    cand = cand.replace(os.sep, "/")
                    tgt = py_index.get(cand)
                else:
                    key = spec.replace(".", "/")
                    tgt = py_index.get(key) or py_index.get(spec)
                    while tgt is None and "/" in key:
                        key = key.rsplit("/", 1)[0]
                        tgt = py_index.get(key)
            elif info["lang"] in ("js", "ts"):
                if spec.startswith("."):
                    tgt = resolve_relative(curdir, spec)
                # bare specifier => external, skip
            if tgt and tgt != rel:
                edges[(rel, tgt)] += 1
    return edges


# ---------------------------------------------------------------------------
# Ranking: PageRank + in-degree
# ---------------------------------------------------------------------------
def pure_pagerank(nodes, out_edges, d=0.85, iters=50):
    n = len(nodes)
    if n == 0:
        return {}
    pr = {u: 1.0 / n for u in nodes}
    for _ in range(iters):
        dangling = sum(pr[u] for u in nodes if not out_edges.get(u))
        newpr = {u: (1 - d) / n + d * dangling / n for u in nodes}
        for u in nodes:
            outs = out_edges.get(u)
            if outs:
                share = d * pr[u] / len(outs)
                for v in outs:
                    newpr[v] += share
        pr = newpr
    return pr


def rank(infos, edges):
    nodes = [i["rel"] for i in infos]
    indeg = Counter()
    out_edges = defaultdict(list)
    for (src, dst), w in edges.items():
        indeg[dst] += w
        out_edges[src].append(dst)

    if HAVE_NX:
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        for (src, dst), w in edges.items():
            g.add_edge(src, dst, weight=w)
        try:
            pr = nx.pagerank(g, alpha=0.85, weight="weight")
        except Exception:
            pr = pure_pagerank(nodes, out_edges)
    else:
        pr = pure_pagerank(nodes, out_edges)
    return pr, indeg


# ---------------------------------------------------------------------------
# Embeddings (optional)
# ---------------------------------------------------------------------------
def doc_for(info):
    path_tokens = re.split(r"[/\\._\-]", info["rel"])
    parts = path_tokens + info["classes"] + [f[0] for f in info["funcs"]]
    if info["summary"]:
        parts.append(info["summary"])
    return " ".join(parts)


def embed(infos, skip_embed):
    docs = [doc_for(i) for i in infos]
    if not skip_embed:
        try:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            vecs = list(model.embed(docs))
            import numpy as _np
            return _np.array(vecs), "fastembed:bge-small-en-v1.5"
        except Exception:
            pass
    if HAVE_SKLEARN:
        try:
            v = TfidfVectorizer(max_features=2000, stop_words="english")
            X = v.fit_transform(docs)
            return X, "sklearn:tfidf"
        except Exception:
            pass
    return None, "none"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster(infos, vecs, method, edges):
    n = len(infos)
    if n == 0:
        return []
    labels = [-1] * n

    if vecs is not None and HAVE_UMAP_HDBSCAN and method == "fastembed:bge-small-en-v1.5":
        try:
            import numpy as _np
            reducer = umap.UMAP(n_neighbors=min(15, max(2, n - 1)), n_components=2,
                                metric="cosine", random_state=42)
            emb2d = reducer.fit_transform(_np.asarray(vecs))
            clu = hdbscan.HDBSCAN(min_cluster_size=max(3, n // 30), min_samples=1)
            labels = clu.fit_predict(emb2d).tolist()
            return _fill_noise(infos, labels, edges)
        except Exception:
            pass

    if vecs is not None and HAVE_SKLEARN:
        try:
            k = max(2, min(14, int(round(n ** 0.5))))
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(vecs).tolist()
            return labels
        except Exception:
            pass

    # Pure fallback (no ML libs): directory-structure clustering. Robust and
    # never collapses the way graph label-propagation does on connected repos.
    return directory_clusters(infos)


def _fill_noise(infos, labels, edges):
    """Assign HDBSCAN noise points (-1) by directory majority."""
    by_dir = defaultdict(Counter)
    for info, lab in zip(infos, labels):
        if lab != -1:
            by_dir[os.path.dirname(info["rel"])][lab] += 1
    out = []
    nextlab = (max(labels) + 1) if labels else 0
    dir_default = {}
    for info, lab in zip(infos, labels):
        if lab != -1:
            out.append(lab)
            continue
        d = os.path.dirname(info["rel"])
        if by_dir[d]:
            out.append(by_dir[d].most_common(1)[0][0])
        else:
            if d not in dir_default:
                dir_default[d] = nextlab
                nextlab += 1
            out.append(dir_default[d])
    return out


def directory_clusters(infos, min_size=3):
    """Group files by directory subtree. Each file is assigned to the deepest
    ancestor directory whose subtree contains at least `min_size` files, so
    small leaf dirs merge with siblings while large dirs stay distinct."""
    counts = Counter()
    for info in infos:
        d = os.path.dirname(info["rel"])
        parts = d.split("/") if d else []
        for k in range(len(parts) + 1):
            counts["/".join(parts[:k])] += 1

    def key_for(rel):
        d = os.path.dirname(rel)
        parts = d.split("/") if d else []
        chosen = ""
        for k in range(len(parts) + 1):
            prefix = "/".join(parts[:k])
            if counts[prefix] >= min_size:
                chosen = prefix
        return chosen or "(root)"

    remap, labels = {}, []
    for info in infos:
        key = key_for(info["rel"])
        if key not in remap:
            remap[key] = len(remap)
        labels.append(remap[key])
    return labels


# ---------------------------------------------------------------------------
# Cluster naming
# ---------------------------------------------------------------------------
def name_cluster(members):
    tokens = Counter()
    for info in members:
        for tok in re.split(r"[/\\._\-]", info["rel"].lower()):
            if len(tok) > 2:
                tokens[tok] += 1
        for sym in info["classes"] + [f[0] for f in info["funcs"]]:
            for tok in TOKEN_RE.findall(sym.lower()):
                if len(tok) > 2:
                    tokens[tok] += 1
    best, best_score = None, 0
    for label, kws in KEYWORD_BUCKETS:
        score = sum(tokens[k] for kw in kws for k in tokens if kw in k)
        if score > best_score:
            best, best_score = label, score
    if best:
        return best
    # else: most common path directory segment
    dirs = Counter(os.path.dirname(i["rel"]).split("/")[-1] or "root" for i in members)
    top = dirs.most_common(1)[0][0]
    return top.replace("_", " ").title() if top else "Misc"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def deep_link(map_url, rel):
    if not map_url:
        return rel
    return f"{map_url.rstrip('/')}/{rel}"


def render(repo, infos, edges, pr, indeg, labels, method, map_url, detail):
    by_rel = {i["rel"]: i for i in infos}
    repo_name = os.path.basename(repo.rstrip("/"))

    # cluster membership
    clusters = defaultdict(list)
    for info, lab in zip(infos, labels):
        clusters[lab].append(info)
    def common_dir(members):
        split = [os.path.dirname(m["rel"]).split("/") for m in members]
        common = []
        for parts in zip(*split):
            if len(set(parts)) == 1 and parts[0]:
                common.append(parts[0])
            else:
                break
        return "/".join(common)

    cluster_name = {}
    base_names = {lab: name_cluster(members) for lab, members in clusters.items()}
    name_collisions = Counter(base_names.values())
    for lab, members in clusters.items():
        nm = base_names[lab]
        if name_collisions[nm] > 1:  # disambiguate by directory, not a counter
            cd = common_dir(members) or "root"
            nm = f"{nm} ({cd})"
        cluster_name[lab] = nm
    rel_cluster = {}
    for info, lab in zip(infos, labels):
        rel_cluster[info["rel"]] = cluster_name[lab]

    # cluster-to-cluster dependency weights
    cross = Counter()
    for (src, dst), w in edges.items():
        cs, cd = rel_cluster.get(src), rel_cluster.get(dst)
        if cs and cd and cs != cd:
            cross[(cs, cd)] += w

    lines = []
    A = lines.append
    A(f"# CODEBASE_NAV — {repo_name}")
    A("")
    A(f"> {len(infos)} files · {len(clusters)} clusters · embed: `{method}` · "
      f"rank: `{'networkx' if HAVE_NX else 'pure'}`")
    A("> Read this file once to navigate without grepping. "
      "`in=` is how many repo files import it.")
    A("")

    # --- Hub files ---
    A("## Hub Files (most imported)")
    hubs = sorted(infos, key=lambda i: (indeg[i["rel"]], pr.get(i["rel"], 0)), reverse=True)
    hub_n = {"min": 8, "std": 12, "full": 20}.get(detail, 12)
    for info in hubs[:hub_n]:
        if indeg[info["rel"]] == 0 and detail != "full":
            break
        rel = info["rel"]
        role = rel_cluster[rel]
        extra = ""
        if info["summary"]:
            extra = f" — {info['summary'][:70]}"
        elif info["classes"]:
            extra = f" — {', '.join(info['classes'][:3])}"
        A(f"- [`{rel}`]({deep_link(map_url, rel)})  in={indeg[rel]}  [{role}]{extra}")
    A("")

    # --- Architecture ---
    A("## Architecture (top dependency routes)")
    if cross:
        for (cs, cd), w in cross.most_common(10):
            A(f"- {cs} → {cd}  (w={w})")
    else:
        A("- (no cross-cluster imports resolved)")
    A("")

    # --- Symbol Index: the token-saver. symbol -> file, no grep needed. ---
    A("## Symbol Index (symbol → file)")
    A("_Jump straight to a definition without searching._")
    sym_entries = []
    for info in infos:
        base = pr.get(info["rel"], 0) + indeg[info["rel"]] * 0.01
        for c in info["classes"]:
            sym_entries.append((c, info["rel"], base + 1.0, "class"))
        for fn, sig in info["funcs"]:
            if fn.startswith("_"):
                continue
            sym_entries.append((fn, info["rel"], base, "fn"))
    sym_entries.sort(key=lambda e: e[2], reverse=True)
    sym_n = {"min": 30, "std": 70, "full": 150}.get(detail, 70)
    per_file = {"min": 2, "std": 3, "full": 6}.get(detail, 3)  # spread across files
    seen, per_file_count, shown = set(), Counter(), 0
    for name, rel, _s, kind in sym_entries:
        if (name, rel) in seen or per_file_count[rel] >= per_file:
            continue
        seen.add((name, rel))
        per_file_count[rel] += 1
        mark = "𝐂" if kind == "class" else "ƒ"
        A(f"- `{name}` {mark} → {rel}")
        shown += 1
        if shown >= sym_n:
            break
    if shown == 0:
        A("- (no notable symbols extracted)")
    A("")

    # --- Clusters ---
    A("## Clusters")
    ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    file_n = {"min": 0, "std": 4, "full": 10}.get(detail, 4)
    cluster_n = {"min": 10, "std": 18, "full": 40}.get(detail, 18)
    hidden = ordered[cluster_n:]
    for lab, members in ordered[:cluster_n]:
        members_sorted = sorted(members, key=lambda i: indeg[i["rel"]], reverse=True)
        hub = members_sorted[0]
        A(f"### {cluster_name[lab]} ({len(members)} files)")
        A(f"Hub: `{hub['rel']}`  (in={indeg[hub['rel']]})")
        # outgoing routes
        outs = [(cd, w) for (cs, cd), w in cross.items() if cs == cluster_name[lab]]
        if outs:
            outs.sort(key=lambda x: x[1], reverse=True)
            A("→: " + ", ".join(f"{cd}({w})" for cd, w in outs[:4]))
        if file_n:
            for info in members_sorted[:file_n]:
                summ = info["summary"][:60] if info["summary"] else (
                    ", ".join(info["classes"][:2]) or f"{len(info['funcs'])} fns")
                A(f"- `{info['rel']}` — {summ}")
        A("")
    if hidden:
        tail = ", ".join(f"{cluster_name[lab]}({len(m)})" for lab, m in hidden)
        A(f"_+{len(hidden)} smaller clusters: {tail}_")
        A("")

    A("---")
    A("_Generated by make_nav.py · regenerate after large refactors._")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate CODEBASE_NAV.md")
    ap.add_argument("--repo", required=True, help="Repository root")
    ap.add_argument("--out", default=None, help="Output directory (default: <repo>_nav/)")
    ap.add_argument("--map-url", default="http://localhost:8000/web/", help="Base URL for deep links ('' to disable)")
    ap.add_argument("--skip-embed", action="store_true", help="Skip neural embeddings (use TF-IDF / graph)")
    ap.add_argument("--detail", choices=["min", "std", "full"], default="std",
                    help="Output richness (min≈5KB, std≈10KB, full≈20KB)")
    ap.add_argument("--max-files", type=int, default=4000, help="Cap files analysed")
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        print(f"error: not a directory: {repo}", file=sys.stderr)
        return 2

    out_dir = args.out or os.path.join(os.getcwd(), os.path.basename(repo.rstrip("/")) + "_nav")
    os.makedirs(out_dir, exist_ok=True)

    repo, rels = discover_files(repo, args.max_files)
    if not rels:
        print("error: no source files found", file=sys.stderr)
        return 1
    print(f"[1/5] discovered {len(rels)} source files", file=sys.stderr)

    infos = []
    for rel in rels:
        lang = LANG_BY_EXT[os.path.splitext(rel)[1].lower()]
        info = parse_file(repo, rel, lang)
        if info:
            infos.append(info)
    print(f"[2/5] parsed symbols & imports", file=sys.stderr)

    edges = build_graph(repo, infos)
    pr, indeg = rank(infos, edges)
    print(f"[3/5] resolved {len(edges)} import edges, ranked", file=sys.stderr)

    vecs, method = embed(infos, args.skip_embed)
    print(f"[4/5] embeddings: {method}", file=sys.stderr)

    labels = cluster(infos, vecs, method, edges)
    md = render(repo, infos, edges, pr, indeg, labels, method, args.map_url, args.detail)

    out_path = os.path.join(out_dir, "CODEBASE_NAV.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    kb = len(md.encode("utf-8")) / 1024
    print(f"[5/5] wrote {out_path}  ({kb:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
