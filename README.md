# codebase-nav

Generate a compact navigation file (`CODEBASE_NAV.md`) for any source repository.
It helps both humans and AI assistants understand a codebase quickly — without reading hundreds of files.

---

## The Problem

When exploring an unfamiliar codebase, you face two options:

- **Read files one by one** — slow, and you often read the wrong ones first
- **Ask an AI** — the AI does the same thing, burning tokens on wrong turns

A 900-file repository has no obvious entry point.
You don't know which files matter, how modules depend on each other, or where the architecture begins.

---

## What CODEBASE_NAV.md Provides

A single file that answers four questions up front:

| Question | Answer |
|----------|--------|
| Which files are most imported? | Hub files ranked by PageRank / in-degree |
| Where is symbol `X` defined? | **Symbol Index** — `X → path/to/file` (no grep) |
| How do modules depend on each other? | Top dependency routes between clusters |
| What does each cluster contain? | Cluster name, size, hub file, connections, summaries |

The **Symbol Index** is the key token-saver: many questions ("where is `UserStore`?",
"which file defines `create_app`?") are answered from the nav file alone — the agent never
reads source or runs a single `grep`.

### Example output

```
## Hub Files (most imported)
- src/lib/stores/index.ts   in=185  [State Management] — global app stores
- backend/open_webui/env.py in=62   [Config/Settings]  — environment config

## Symbol Index (symbol → file)
- `UserStore`   𝐂 → src/lib/stores/user.ts
- `create_app`  ƒ → backend/open_webui/main.py
- `embed_query` ƒ → backend/open_webui/retrieval/utils.py

## Architecture (top dependency routes)
- UI Components → Chat/LLM   (w=221)
- API/Server    → Search/RAG (w=199)

## Clusters
### Authentication (10 files)
Hub: backend/open_webui/models/auths.py  (in=43)
→: API/Server(3)
- backend/open_webui/models/auths.py — auth models & token logic
```

---

## How It Works

```
Source files (Python · JS/TS · Go · Java · Ruby · Rust)
    ↓  regex extraction (functions + signatures, imports, classes, exports, docstrings)
Embedding (fastembed BAAI/bge-small-en-v1.5 · fallback: TF-IDF · fallback: none)
    ↓
UMAP → 2D + HDBSCAN clustering  (fallback: KMeans · fallback: directory structure)
    ↓
Import resolution → PageRank / in-degree  (networkx · fallback: pure-Python)
    ↓
CODEBASE_NAV.md  (Hub Files · Symbol Index · Architecture · Clusters)
```

No build tools, no external services. **Runs on the Python standard library alone** —
every heavy dependency is optional and auto-detected. Installing them only improves
clustering quality.

---

## Usage

### Command line

```bash
python3 map/pipeline/make_nav.py --repo /path/to/any-repo
```

Options:
```
--repo       PATH    Repository root (required)
--out        DIR     Output directory (default: <repo_name>_nav/)
--detail     LEVEL   min | std | full  (default: std)
--map-url    URL     Base URL for deep links ('' to disable)
--skip-embed         Skip neural embeddings (use TF-IDF / directory clustering)
--max-files  N       Cap files analysed (default: 4000)
```

`--detail` controls the richness/size tradeoff:

| Level  | Approx size | Symbol Index | Per-cluster file lists |
|--------|-------------|--------------|------------------------|
| `min`  | ~6 KB       | 30 symbols   | no                     |
| `std`  | ~10–15 KB   | 70 symbols   | yes (4/cluster)        |
| `full` | ~40 KB      | 150 symbols  | yes (10/cluster)       |

(Sizes scale with repo size; figures are for a ~1,200-file repository.)

### Claude Code skill

```
/codebase-nav /path/to/any-repo
```

Claude runs the pipeline, reads the output, and explains the architecture.

---

## Effect on AI-assisted development

The win is not a fixed per-question discount — it's **eliminating the wrong-turn read loop**,
and that scales with repo size.

| Scenario (900-file repo) | Without nav | With nav |
|--------------------------|------------|----------|
| "Where is `create_app` defined?" | grep → open 2–4 candidates | Symbol Index → exact file, **0 source reads** |
| "Explain the auth system" | grep → wrong files → retry → right files (~10–40K tokens) | jump to Auth cluster hub directly (~2–4K tokens) |
| "What's the biggest dependency?" | unknowable without analysis | `stores/index.ts` — imported by 185 files |

**Where the tokens actually go:**

- *Without nav*, the dominant cost on an unfamiliar repo is **exploratory misses** —
  globbing, grepping, and reading files that turn out to be irrelevant. On a large repo a
  single architectural question easily costs 10K–40K tokens before the agent finds the right file.
- *With nav*, the agent reads the nav file **once** (~3–5K tokens for `std`), then goes
  straight to the right file. For "where is X" questions answered by the Symbol Index, it reads
  **no source at all**.

So the realistic saving on a large repo is closer to **5–10×** on exploration-heavy questions,
not a flat ~40% — and it compounds because the nav is read once and reused across every
subsequent question in the session (and stays in the prompt cache).

> Honest caveat: these are estimates based on typical exploration patterns, not a controlled
> benchmark. The exact saving depends on repo size, question type, and how lost the agent would
> otherwise get. On a tiny repo the nav barely pays for itself; on a large one it pays for itself
> on the first question.

---

## Requirements

Works out of the box with **Python 3.10+ and nothing else.** For higher-quality semantic
clustering, optionally install:

```bash
pip install fastembed umap-learn hdbscan networkx scikit-learn numpy
```

No GPU required · Internet needed only for the first embedding-model download (~25 MB),
and only if neural embeddings are enabled.

---

---

# codebase-nav（日本語）

任意のソースリポジトリに対して、コンパクトなナビゲーションファイル（`CODEBASE_NAV.md`）を生成します。
人間と AI アシスタントの両方が、大量のファイルを読まずにコードベースを素早く把握できるようにします。

---

## 問題

見慣れないコードベースを探索するとき、選択肢は2つしかありません。

- **ファイルを一つずつ読む** — 遅い、しかも最初に読むファイルを間違えることが多い
- **AI に聞く** — AI も同じことをやるので、トークンを間違い探しに消費する

900 ファイルあるリポジトリに明確な入り口はありません。
どのファイルが重要か、モジュールがどう依存しているか、アーキテクチャがどこから始まるかが分からない。

---

## CODEBASE_NAV.md が答える問い

1枚のファイルで以下が分かります。

| 問い | 答え |
|------|------|
| 最も import されているファイルは？ | PageRank / 被参照数でランク付けした Hub ファイル |
| `X` の定義はどこ？ | **Symbol Index** — `X → path/to/file`（grep 不要） |
| モジュール間の依存関係は？ | クラスタ間の主要な依存ルート |
| 各クラスタには何が入っている？ | クラスタ名・サイズ・Hub ファイル・接続先・要約 |

**Symbol Index** が最大のトークン削減ポイントです。「`UserStore` はどこ？」「`create_app` を定義してるファイルは？」
といった質問は、ナビファイルだけで答えられます（ソースを読まない・grep もしない）。

---

## 仕組み

```
ソースファイル（Python · JS/TS · Go · Java · Ruby · Rust）
    ↓  正規表現で関数（シグネチャ付）・import・クラス・export・docstring を抽出
Embedding（fastembed BAAI/bge-small-en-v1.5 · フォールバック: TF-IDF · なし）
    ↓
UMAP → 2D + HDBSCAN クラスタリング（フォールバック: KMeans · ディレクトリ構造）
    ↓
import 解決 → PageRank / 被参照数（networkx · フォールバック: pure-Python）
    ↓
CODEBASE_NAV.md（Hub Files · Symbol Index · Architecture · Clusters）
```

ビルドツール不要、外部サービス不要。**Python 標準ライブラリだけで動作します。**
重いライブラリはすべて任意で、入っていれば自動検出してクラスタの質を上げます。

---

## 使い方

### コマンドライン

```bash
python3 map/pipeline/make_nav.py --repo /path/to/any-repo [--detail std]
```

`--detail` は情報量とサイズのトレードオフ（min ≈6KB / std ≈10-15KB / full ≈40KB、~1,200 ファイル時）。

### Claude Code スキル

```
/codebase-nav /path/to/any-repo
```

Claude がパイプラインを実行し、出力を読んでアーキテクチャを説明します。

---

## AI 支援開発への効果

効果は「1問あたり固定で◯%削減」ではなく、**「外れファイルを読むループ」を消すこと**で、
リポジトリが大きいほど効きます。

| シナリオ（900 ファイル） | ナビなし | ナビあり |
|---------|---------|---------|
| 「`create_app` の定義はどこ？」 | grep → 候補を2〜4本開く | Symbol Index → 一発、**ソース読込ゼロ** |
| 「認証の仕組みを説明して」 | grep → 外れ → やり直し → 当たり（約1〜4万トークン） | Auth クラスタの Hub に直行（約2〜4千トークン） |
| 「最も依存されるファイルは？」 | 解析しないと不明 | `stores/index.ts` — 185 ファイルから参照 |

**トークンはどこで消えるか:**

- *ナビなし*の支配的コストは**外れ探索**（glob/grep/無関係ファイルの読込）。大規模リポジトリでは
  1つのアーキテクチャ質問で当たりに辿り着くまで簡単に 1〜4 万トークン使う。
- *ナビあり*はナビを**1回**読むだけ（std で約3〜5千トークン）で当たりに直行。Symbol Index で答えられる
  質問なら**ソースを一切読まない**。

つまり大規模リポジトリでの現実的な削減は、探索が重い質問で**5〜10倍**になり得ます（一律40%ではない）。
しかもナビは1回読めば以降の全質問で使い回せる（prompt cache にも乗る）ので効果は累積します。

> 正直な注意: これは実測ベンチではなく、典型的な探索パターンからの見積もりです。実際の削減量は
> リポジトリ規模・質問の種類・AI がどれだけ迷うかに依存します。小さいリポジトリでは元が取れにくく、
> 大きいリポジトリでは1問目で元が取れます。

---

## 必要なパッケージ

**Python 3.10 以上だけで動きます。** 意味ベースのクラスタリングの質を上げたい場合のみ、任意で:

```bash
pip install fastembed umap-learn hdbscan networkx scikit-learn numpy
```

GPU 不要 · インターネット接続は Embedding 有効時の初回モデルダウンロード（約25MB）のみ。
