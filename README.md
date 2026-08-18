# Milvus 3.0 and Elasticsearch 9.4: a side-by-side query semantics lab

A simple, honest lab framework for evaluating query and search semantics
between Milvus 3.0 and Elasticsearch 9.4.

*Used as the basis for: [https://talks.simonhearne.com/2026-08-milvus-search-gap/](https://talks.simonhearne.com/2026-08-milvus-search-gap/)*

One `docker compose up` brings up both engines and a browser app in which each
search capability is a panel. The Milvus and Elasticsearch calls sit next to
each other and are **editable and executable**. The ~98k-row `amazon_reviews`
dataset is a single Parquet file (fetched in setup, below) that loads
byte-identically into both engines, so the same floats are searched on each
side.

Both engines run under the same caps (8 GB / 4 CPU each), same dataset, same
1024-d COSINE vectors, and matched BM25 analyzer chains (close, not exact; see
*Known gaps*). The first tab (**Introduction**) validates all of this live
against the running containers with green/amber checks.

## What's in the box

| Service | Port | Purpose |
|---|---|---|
| Milvus 3.0 standalone | 19530, 9091 | vector engine + WebUI (`/webui/`) |
| Elasticsearch 9.4 | 9200 | vector engine (basic licence, security disabled; local lab only) |
| Lab app | 8080 | the side-by-side comparison UI (localhost only) |

`etcd` and `minio` also come up; they are Milvus's storage dependencies.

Ports are defaults; override any of them in `.env` (`MILVUS_PORT`,
`MILVUS_METRICS_PORT`, `ES_PORT`, `DEMO_PORT`) if another stack on the machine
holds one. Containers are namespaced as `mvs-es-lab-<service>-1`, so other
Milvus/ES compose stacks can run alongside this one.

## Prerequisites

- Docker with **16 GB** allocated. The 8 GB per-engine limits are ceilings, not
  reservations.
- Python 3 on the host is optional. Every script can run inside the `demo`
  container instead, and `data/fetch.py` needs only the standard library.

## Setup

```bash
cp .env.example .env
docker compose up -d          # boots both engines + the lab app
```

### 1. Fetch the vectors (~370 MB, once)

```bash
python data/fetch.py          # or: docker compose exec demo python data/fetch.py
```

The Parquet file lives in a
public Hugging Face dataset:
[`simonhearne/milvus-es-live-data`](https://huggingface.co/datasets/simonhearne/milvus-es-live-data).
The script is a no-op once the file is present, downloads via a `.part` rename
so an interrupted fetch can't leave a truncated file, and `--verify` checks the
sha256. Everything else the loaders need (`data/schema.json`) is in the repo.

### 2. Load both engines

Loaders run inside the `demo` container:

```bash
docker compose exec demo python data/load_milvus.py
docker compose exec demo python data/load_es.py
# optional small isolated bilingual (EN+ZH) store for the Multilingual panel
docker compose exec demo python data/load_multilingual.py
```

The bundled ES image already includes the ICU plugin (needed for the
multilingual panel).

### 3. Build ground truth (for the recall numbers)

```bash
docker compose exec demo python bench/ground_truth.py   # -> bench/gt.npz
```

### 4. Open the lab

http://localhost:8080

## The lab app

Each capability is a panel with the Milvus and ES calls side by side. Choose a
source product / vector if required, optionally change the run count, hit
**Run**, and you get median/min/max plus a strip plot of every run, recall and
cross-engine agreement. The hits are rendered as result cards, and the raw
responses are available underneath.

**The app is bound to `127.0.0.1`.** It executes arbitrary Python from the
browser against your stack; that is what makes the snippets editable. Do not
widen the port mapping in `docker-compose.yml`, and do not run it anywhere but
a local machine.

Seed code comes from `webapp/snippets.py`, the single source of truth for every
panel. Snippets are executable: they must end in an expression (the raw SDK
response).

### The sixteen panels

| panel | what it shows |
|---|---|
| `dense` | Dense ANN search: RaBitQ+SQ8 vs `bbq_hnsw`, cosine, top-10 |
| `filtered` | Selective scalar filter applied to the vector search |
| `bm25` | BM25 lexical search, same nominal analyzer chain both sides |
| `synonyms` | Query `tablet` → iPad results; Milvus 3.0 synonym filter vs ES synonym analyzer |
| `multilingual` | Chinese query → jieba tokens; Milvus `language_identifier` auto-detects vs ES ICU analyzer |
| `hybrid` | Dense + lexical with reciprocal-rank fusion; licence-gated on the ES side, see below |
| `grouped` | Grouping / collapse: one representative hit per `main_category` |
| `image` | Image similarity: a second dense vector in the same collection |
| `ordered` | Kernel-side ordering: retrieve by similarity, present ordered by scalar fields |
| `aggregate` | Server-side aggregation: `count`/`avg`/`min`/`max` pushed into the kernel |
| `rescore` | Milvus `FunctionScore` + decay reranker vs ES `function_score` + `gauss`: same Gaussian-decay vocabulary (`origin`/`scale`/`offset`/`decay`), cheaper products favoured on both sides |
| `highlight` | Milvus `LexicalHighlighter` vs ES `highlight`: same parameters (`pre_tags`/`post_tags`/`fragment_size`/`num_of_fragments`), same response shape, rendered on the result cards |
| `phrase` | Milvus `PHRASE_MATCH(...)` filter (BM25 still scores) vs ES `match_phrase` with `slop`; see the badge note below |
| `geo` | Milvus `st_dwithin` on a `GEOMETRY` field vs ES `geo_distance`: same radius query, opposite axis order (`POINT(lon lat)` vs `{lat, lon}`), a common migration bug made visible |
| `dates` | Milvus ISO 8601 literal filter vs ES's `now-30d/d`: equivalent date math, but the client supplies the clock (see *Known gaps*) |
| `typo` | Two mitigations for misspelled queries, each covering a different case; see the badge note below |

Two panels have result badges that are easy to misread:

- **`phrase` reads `overlap 2/10` and that is correct.** The engines agree on
  the *set*: 683 of the 686 matching docs, 99.6%, but rank that shared set by
  different things (BM25 vs phrase-proximity scoring). The top-10s are
  different documents, not the same ten reordered. It is not a set disagreement.
- **`typo` opens on a *known* misspelling** (`"laptp charjer"`, enumerated in
  `data/synonyms.txt`): the synonym filter returns a full 10 hits where plain
  BM25 returns 0 on either engine. An unenumerated misspelling returns 0 on
  both sides.

### Hybrid fusion is licence-gated on Elasticsearch

The authored ES snippet on the `hybrid` panel uses the `rrf` retriever. Run
against this stack's basic-licence Elasticsearch it returns a live **403**: the
`rrf` and `linear` retrievers are Platinum+ features. Milvus's `RRFRanker` and
`WeightedRanker` are in the open-source build.

A button on the ES pane header, **Rewrite for basic licence**, swaps in a
free-tier rewrite that fuses `knn` + `match` with hand-tuned `boost` weights,
which runs green on both engines. Hand-tuning is what the free path costs: BM25
scores are unbounded while cosine similarity sits between 0 and 1, which is
exactly the normalisation problem RRF resolves.

### What the numbers will and won't tell you

- **Latency** is single-machine Docker behaviour, not a benchmark (see *Reading
  the numbers honestly*).
- **Recall** is real and reproduces: it is scored against exact ground truth in
  the matching vector space. It is *suppressed* if you edit a snippet so it no
  longer searches a bound `qv` / `img_qv`.
- Use the product drop-down to change the source query vector (200 ground-truth
  queries), not the editor.

## The dataset

`amazon_reviews` (Amazon product metadata, one row per `parent_asin`, 97,894
rows):

- **PK**: `parent_asin` (VarChar)
- **Dense**: `text_vec` and `image_vec`, both 1024-d, COSINE (enables the
  cross-modal / image-search panel)
- **BM25**: `text_snippet` (analyzer) → `text_snippet_bm25` function →
  `text_sparse` (BM25 index). `text_sparse` is a function output, regenerated
  on local insert and not carried as data in the Parquet. The ES side uses a
  matching `amazon_en` analyzer (standard + lowercase + asciifolding + english
  stemmer + english stopwords) so lexical scoring is close to apples-to-apples,
  with the caveats in *Known gaps*.
- **Array**: `categories` (Array&lt;VarChar&gt; → ES keyword array)
- **Scalars**: `title`, `store`, `main_category`, `price`, `average_rating`,
  `rating_number`, `image_url`
- **Synthetic**: `store_location` (GEOMETRY / `geo_point`) and `first_seen`
  (`TIMESTAMPTZ` / ES `date`), see below. `first_seen` is a real timestamp
  type, not an integer: it is written and read back as RFC3339
  (`2024-08-07T14:25:04Z`) and filtered with ISO 8601 literals
  (`first_seen > ISO '2026-07-01T00:00:00Z'`).

### Synthetic columns: `store_location` and `first_seen`

The Amazon product metadata has neither coordinates nor listing dates. Anyone
migrating a real product search off Elasticsearch has both, so
`data/synthesize.py` manufactures them at load time rather than editing the
Parquet:

- **`store_location`** is derived from `store` (sha256-seeded, so the same
  store always lands in the same one of ~40 world cities). Sellers share a
  location on purpose: a radius query returns a coherent set instead of noise.
- **`first_seen`** is a listing date seeded per `parent_asin`, with ~20% of the
  corpus landing inside the trailing 30 days.

**Operational trap:** `first_seen` is anchored to *load time*, not a fixed
calendar date. The `dates` panel filters on ES's `now-30d/d`, and a corpus
pinned to a hardcoded day would quietly stop returning anything once real time
moved past it. The consequence: **if the corpus was loaded more than 30 days
ago, the `dates` panel returns nothing on both engines.** Reload
(`data/load_milvus.py` + `data/load_es.py`) to re-anchor `first_seen`; this is
also why the app's result cache is re-cut after every reload.

## Known gaps (both sides of the story)

- **No edit-distance fuzziness in Milvus 3.0.** The `typo` panel's mitigation,
  enumerating known misspellings in the synonym filter (`data/synonyms.txt`),
  is a real workaround, not a substitute; it only ever catches typos someone
  already wrote down. An ES user could adopt the same mitigation, which is why
  both sides are shown.
- **No server-side `now` in Milvus 3.0.** Narrower than it first looks. Milvus
  *does* have date math on `TIMESTAMPTZ`: ISO 8601 literals
  (`first_seen > ISO '2026-07-01T00:00:00Z'`) and ISO 8601 duration arithmetic
  (`first_seen + INTERVAL 'P30D'`), both usable inside a vector-search filter.
  What has no equivalent is ES's server-side clock: there is no `now()` or
  `CURRENT_TIMESTAMP`, so a relative window like `now-30d/d` has to be anchored
  by the client and interpolated into the filter. There is also no component
  extraction: `EXTRACT(YEAR FROM …)` is rejected. Worth knowing: the
  ISO-literal comparison is the fast form (~4 ms, matching an unfiltered
  search), while `INTERVAL` arithmetic costs ~9 ms because the per-row
  computation defeats index pushdown; the panel uses the literal and mentions
  the interval form in a comment.
- **ES hybrid fusion is a paid feature.** The `rrf` and `linear` retrievers
  require a Platinum+ licence; on the basic licence you hand-tune `boost`
  weights (see *Hybrid fusion is licence-gated on Elasticsearch*). Milvus's
  `RRFRanker` and `WeightedRanker` are in the open-source build.
- **Analyzer parity is close, not exact.** Both sides run `standard` +
  lowercase + asciifolding + english stemmer + english stopwords, but the two
  `standard` tokenizers disagree on tokens containing digits and punctuation
  (`5.0`, `1,000`, `amazon.com`), and `language: "english"` resolves to Porter1
  in Elasticsearch and to Porter2 in Milvus.
- **`SemanticHighlighter` exists but is not covered here.** Milvus 3.0 also
  ships embedding-based highlighting (no ES equivalent; ES highlighting is
  lexical only). It isn't in the `highlight` panel because it needs a model
  deployment this offline lab has no way to provide.

## Reading the numbers honestly

A single-machine Docker stack on a laptop **is not a benchmark**. Latency here
is dominated by JVM warmup, container CPU limits, and cold cache.

- For **performance claims**, cite published large-scale benchmarks rather than
  this environment.
- What this environment does support: **feature coverage**, **query semantics**,
  **implementation ergonomics**, and **recall**, which is architectural and
  reproduces honestly.

**Recall is params-dependent**. Two knobs move independently, and it is
**`refine_k`, not `nprobe`**, that carries the recall here:

| `nprobe` | `refine_k` | Milvus recall@10 | Milvus latency |
|---|---|---|---|
| unset | 1.0 | 0.928 | 1.3 ms |
| 64 | 1.0 | 0.930 | 3.1 ms |
| unset | 2.0 | 0.980 | 1.2 ms |
| **64 (default)** | **2.0** | **0.990** | **3.1 ms** |
| 256 | 2.0 | 0.993 | 9.6 ms |
| 1024 | 2.0 | 0.993 | 33.9 ms |

Read the first two rows against each other: at `refine_k=1.0`, adding
`nprobe=64` is worth **+0.003** recall. Now read rows 1→3 and 2→4: raising
`refine_k` 1.0 → 2.0 is worth **+0.05 with `nprobe` unset, +0.06 at
`nprobe=64`**. The committed default sets both, and essentially all of the gain
is `refine_k`'s.

`nprobe` does bite, but only far below the default. At `refine_k=2.0`:
`nprobe=1` → 0.788, `2` → 0.898, `4` → 0.933, `8` → 0.980, `16` → 0.983,
`64` → 0.990. It has saturated by ~8.

## Operational notes

**Milvus image tag** is pinned to `MILVUS_TAG=v3.0.0` (GA). To move to another
tag, change that line in `.env` and:

```bash
docker compose up -d --force-recreate milvus
```

The same default is repeated as a `:-` fallback in `docker-compose.yml` (twice)
and as `MILVUS_DEFAULT_TAG` in `webapp/environment.py`, for when `.env` is
absent; keep the three in sync. Back up `volumes/` before any tag change: there
is no rollback path if a tag changes the on-disk format.

**The `demo` container declares no Docker healthcheck.** Trust the in-app
health dots and `curl localhost:8080/api/health`, not Docker's status column.

## Verifying the environment

The UI has no automated end-to-end tests. This checklist is the test:

1. `docker compose up -d` and wait for both in-app health dots to go green.
2. Run every panel once at the default run count. All must return without an
   error box, except `hybrid`, whose ES pane returns a licence 403 by design
   (see above).
3. Run `dense` a second time and confirm the median drops. That is warm cache;
   first-run numbers are cold-JVM and cold-cache, and are not comparable with
   warm ones.
4. Check `image` renders actual product images. If they're missing, `image_url`
   didn't hydrate.
5. Exercise the degraded path: `docker compose stop elasticsearch`, hit Run,
   confirm Milvus still renders and agreement reads `n/a`. Hit **Show last good
   run** and confirm the CACHED stamp. `docker compose start elasticsearch`.
6. Confirm every asset is same-origin in DevTools → Network. The lab carries no
   third-party assets and is designed to run fully offline.
7. If it has been more than 30 days since the last load, reload both engines,
   otherwise the `dates` panel returns nothing (see *Synthetic columns*).
8. Choose a run count. 10 is the default; 20 resolves the distribution better
   at twice the wall clock.

## License

Code is [MIT](LICENSE). The vendored CodeMirror files under
`webapp/static/vendor/` are MIT-licensed by their authors (headers retained).
The product metadata derives from the
[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
dataset (McAuley Lab, UC San Diego), redistributed as a prepared Parquet with
precomputed embeddings at
[`simonhearne/milvus-es-live-data`](https://huggingface.co/datasets/simonhearne/milvus-es-live-data).
