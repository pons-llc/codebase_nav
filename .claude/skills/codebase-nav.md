---
name: codebase-nav
description: Generate CODEBASE_NAV.md for any repository to enable fast, token-efficient codebase navigation
---

This skill generates a `CODEBASE_NAV.md` for any target repository.
In addition to Hub files, dependency routes, and clusters, it emits a
**Symbol Index (symbol → file)**, so "where is X defined?" questions can be
answered without grepping or reading source.

## Steps

1. Read the repository path from the user's message.
   If none is given, ask: "Which repository should I target? (please give an absolute path)".

2. Decide the output location (default to `<repo_name>_nav/` in the current directory).

3. Run the following command (any Python 3.10+ works; runs on the standard library alone):
   ```bash
   python3 /Users/tatsurohatori/Documents/codeMapping/map/pipeline/make_nav.py \
     --repo <repo_path> \
     --out <out_dir> \
     --detail std
   ```
   - `--detail` is `min` (~6KB) / `std` (~10-15KB) / `full` (~40KB) — the size/richness tradeoff.
   - If fastembed / umap-learn / hdbscan are installed, semantic clustering is used automatically.
     Otherwise it falls back TF-IDF → directory-structure clustering (use `--skip-embed` to skip explicitly).
   - Only the first run with neural embeddings downloads the model (1–2 min).

4. Read the generated `CODEBASE_NAV.md` with the Read tool and report:
   - File count, cluster count, and the embed/rank backends used (shown in the header line).
   - Main dependency routes (Architecture section).
   - Top 3 Hub files and their roles.
   - That the Symbol Index is queryable (e.g. which file a representative class lives in).

## Notes

- `make_nav.py` is self-contained (no external file dependencies; heavy libraries are optional).
- Backends degrade in tiers:
  - Embedding: fastembed → sklearn TF-IDF → none
  - Clustering: UMAP+HDBSCAN → sklearn KMeans → directory structure (pure-Python)
  - Ranking: networkx PageRank → pure-Python PageRank
  Only suggest `pip install fastembed umap-learn hdbscan networkx scikit-learn` if cluster quality matters.
- Supported languages: Python / JS / TS / Go / Java / Ruby / Rust (import resolution is most accurate for Python and JS/TS).
- Output size scales with `--detail` and repo size. Tune the analysis cap with `--max-files`.
- `--map-url` changes the links inside the nav (default: http://localhost:8000/web/, `''` to disable).
