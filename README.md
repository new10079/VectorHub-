# VectorHub — Lightweight Vector Database

A lightweight Python + C++ hybrid vector database with HNSW indexing.
**Phase 2** adds true HNSW graph search, cosine distance, persistence, metadata,
delete/update, and a benchmark harness. **Phase 3** adds an end-to-end RAG
(Retrieval-Augmented Generation) demo built entirely on top of the existing
Phase 1/2 API.

> **Positioning**: lighter than Milvus, easier than Faiss.

## Features

| | Phase 1 | Phase 2 | Phase 3 (current) |
|---|---|---|---|
| C++ HNSW index | brute-force | true HNSW graph | — |
| Distance metrics | L2 only | **L2 + cosine** | — |
| HNSW params (M, ef) | — | ✓ | — |
| Persistence (save/load) | — | ✓ | ✓ (used by the RAG demo) |
| Metadata store | — | ✓ | ✓ (used to store source/chunk/text) |
| Metadata filter | — | ✓ (equality) | ✓ (used for per-source retrieval) |
| delete / update | — | ✓ | — |
| brute_force_search | — | ✓ (recall testing) | — |
| Benchmark script | — | ✓ | — |
| Text loading / chunking | — | — | ✓ (`vectorhub.rag`) |
| RAG demo (`examples/rag_demo.py`) | — | — | ✓ |

## Installation

### Requirements

- Python 3.8+
- CMake 3.15+ (`pip install cmake`)
- g++ 7+ (MSYS2/ucrt64 on Windows, GCC/Clang on Linux/macOS)
- pybind11, ninja (`pip install pybind11 ninja`)
- NumPy
- Network access on first configure, to fetch [xsimd](https://github.com/xtensor-stack/xsimd) (header-only SIMD library, pulled in automatically via CMake `FetchContent`; no extra install step)

### Build

```bash
pip install cmake pybind11 ninja scikit-build-core numpy pytest

# Build the C++ extension
mkdir build_cpp && cd build_cpp
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build . --config Release
cd ..

# Copy .pyd into the package
cp build_cpp/_vectorhub*.pyd vectorhub/

# On Windows with MSYS2, also copy runtime DLLs:
# cp D:/msys2/ucrt64/bin/libgcc_s_seh-1.dll vectorhub/
# cp D:/msys2/ucrt64/bin/libwinpthread-1.dll vectorhub/
# cp D:/msys2/ucrt64/bin/libstdc++-6.dll     vectorhub/

pip install -e .
```

### Verify

```bash
python -c "from vectorhub import VectorCollection; print('OK')"
python examples/simple_demo.py
pytest
```

## Quick Start

```python
from vectorhub import VectorCollection

db = VectorCollection(dim=128, metric="l2", M=16, ef_construction=200, ef_search=50)

db.add(vectors=my_vectors, ids=my_ids)

results = db.search(query_vector, k=10)
for id_, dist in results:
    print(id_, dist)
```

## API Reference

### `VectorCollection(dim, metric="l2", M=16, ef_construction=200, ef_search=50)`

| Param | Default | Meaning |
|---|---|---|
| `dim` | required | Vector dimensionality |
| `metric` | `"l2"` | `"l2"` (Euclidean) or `"cosine"` |
| `M` | 16 | Max neighbours per HNSW node per layer. Higher → better recall, more RAM |
| `ef_construction` | 200 | Candidate pool during build. Higher → better graph, slower inserts |
| `ef_search` | 50 | Candidate pool during search. Higher → better recall, slower queries |

### `add(vectors, ids, metadatas=None)`

Add vectors. `metadatas` is an optional `list[dict]` of equal length.

```python
db.add(
    vectors=[[0.1, 0.2, ...], ...],
    ids=[1, 2, 3],
    metadatas=[{"source": "wiki", "year": 2024}, {"source": "news"}, None],
)
```

Duplicate IDs raise `RuntimeError`. Mismatched dimensions raise `ValueError`.

### `search(query, k=10, filter=None, include_metadata=False)`

Returns `list[(id, distance)]` by default.

```python
# Basic search
results = db.search(query, k=5)

# With metadata returned
results = db.search(query, k=5, include_metadata=True)
# → [(id, distance, {"source": "wiki"}), ...]

# Equality filter on metadata
results = db.search(query, k=5, filter={"source": "wiki"})
```

### `brute_force_search(query, k=10)`

Exact linear-scan search. Use for recall benchmarking or ground-truth generation.

### `delete(id)`

Lazy-delete a vector. Deleted IDs are excluded from future searches.

```python
db.delete(42)
```

Raises `KeyError` if the ID does not exist or is already deleted.

### `update(id, vector, metadata=None)`

Delete + re-insert. Graph edges are updated.

```python
db.update(42, new_vector, metadata={"version": 2})
```

### `save(path)` / `VectorCollection.load(path)`

Persist to two files: `<path>` (binary index) and `<path>.meta.json`.

```python
db.save("/tmp/my_index.vhb")

db2 = VectorCollection.load("/tmp/my_index.vhb")
# db2 is ready to search and add more vectors
```

### `set_ef_search(ef)`

Adjust search quality at runtime without rebuilding.

```python
db.set_ef_search(200)  # higher recall, slower
db.set_ef_search(20)   # faster, lower recall
```

### `len(db)`

Returns the number of live (non-deleted) vectors.

## HNSW Parameter Guide

| Scenario | M | ef_construction | ef_search |
|---|---|---|---|
| Fast prototyping (small dataset) | 8 | 50 | 20 |
| Default (good balance) | 16 | 200 | 50 |
| High recall (production) | 32 | 400 | 100 |
| Maximum recall | 48 | 500 | 200+ |

**Rule of thumb**: increase `ef_search` first (cheapest), then `M` if recall is still insufficient.

## RAG Demo (Phase 3)

`examples/rag_demo.py` demonstrates a full retrieval-augmented generation
pipeline built entirely on the existing `VectorCollection` API — no changes
to the C++ core were needed.

Pipeline: **load** a document → **chunk** it → **embed** each chunk → **index**
into a `VectorCollection` (with `source` / `chunk_id` / `text` metadata) →
**retrieve** top-k chunks for a question → **generate** an answer.

### Install the extra dependency

Embeddings are produced with [sentence-transformers](https://www.sbert.net/),
which is an *optional* dependency (not required to use core VectorHub):

```bash
pip install sentence-transformers
# or: pip install vectorhub[rag]
```

Optional PDF loading:

```bash
pip install pypdf
# or: pip install vectorhub[pdf]
```

If the model can't be downloaded (no network / corrupted cache), both
`vectorhub.rag.get_default_embedder()` and the demo script raise a clear
error explaining what to do (check network, pre-download the model, or set
`HF_HUB_OFFLINE=1` if it's already cached).

> If `huggingface.co` is unreachable from your network (common in mainland
> China), set `HF_ENDPOINT=https://hf-mirror.com` before running the demo —
> this is what was used to validate the demo end-to-end in this repo.

### Run it

```bash
python examples/rag_demo.py \
    --data examples/data/sample.txt \
    --query "What is VectorHub?" \
    --top-k 3 \
    --llm mock
```

With persistence (reuses the Phase 2 `save()`/`load()` API — the index is
built once, then reloaded from disk on subsequent runs instead of
re-embedding):

```bash
python examples/rag_demo.py \
    --data examples/data/sample.txt \
    --query "What is VectorHub?" \
    --top-k 3 \
    --llm mock \
    --persist-path examples/data/vectorhub_index.bin
```

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--data` | required | Path to a `.txt` file (`.pdf` supported with `pypdf` installed) |
| `--query` | required | Question to ask |
| `--top-k` | 3 | Number of chunks to retrieve |
| `--chunk-size` | 256 | Chunk size in characters |
| `--overlap` | 32 | Overlap between consecutive chunks, in characters |
| `--llm` | `mock` | `mock` (offline, no dependency), `openai`, or `deepseek` (via LangChain) |
| `--embedding-model` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `--persist-path` | none | If set, saves the index there and reloads it on the next run |

### `mock` vs `openai` vs `deepseek` answer generation

- **`mock` (default)** — no external API call. It formats the retrieved
  chunks into a readable answer stub. This is what makes the demo runnable
  with zero API keys and zero network access (beyond the one-time embedding
  model download).
- **`openai`** — calls the OpenAI chat completion API. Requires the optional
  `openai` package (`pip install openai` / `pip install vectorhub[openai]`)
  and an API key, either passed as `api_key=` to
  `vectorhub.rag.openai_generate_answer()` or via the `OPENAI_API_KEY`
  environment variable. If no key is set, it raises a `RuntimeError` telling
  you to use `--llm mock` instead — it never silently falls back.
- **`deepseek`** — calls the DeepSeek chat API through the LangChain
  framework (`langchain_deepseek.ChatDeepSeek`). Requires the optional
  `langchain-deepseek` package (`pip install langchain-deepseek` / `pip
  install vectorhub[deepseek]`) and an API key, either passed as `api_key=`
  to `vectorhub.rag.deepseek_generate_answer()` or via the
  `DEEPSEEK_API_KEY` environment variable. Same no-silent-fallback behavior
  as `openai`.

`examples/rag_demo.py` also auto-loads a repo-root `.env` file (via the
optional `python-dotenv` package — `pip install python-dotenv` / `pip
install vectorhub[dotenv]`), so `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` set
there are picked up automatically without exporting them in the shell. If
`python-dotenv` isn't installed, this is silently skipped and env vars set
some other way still work.

### `vectorhub.rag` module

Reusable building blocks (each independently testable, none of them touch
the C++ core):

```python
from vectorhub.rag import (
    load_document,       # .txt (always) / .pdf (needs pypdf)
    chunk_text,           # text -> list[{chunk_id, source, text}]
    get_default_embedder, # -> callable(list[str]) -> list[list[float]]
    index_chunks,         # chunks + embed_fn -> inserts into a VectorCollection
    retrieve,             # query + embed_fn -> ranked list of chunk dicts
    build_prompt,
    mock_generate_answer,
    openai_generate_answer,
    deepseek_generate_answer,
    generate_answer,       # dispatches on mode="mock" | "openai" | "deepseek"
)
```

`chunk_text` and `mock_generate_answer` have no dependencies at all;
`get_default_embedder` needs `sentence-transformers`; `openai_generate_answer`
needs `openai` + an API key; `deepseek_generate_answer` needs
`langchain-deepseek` (+ `langchain-core`) and a `DEEPSEEK_API_KEY`.
`tests/test_rag_workflow.py` exercises the whole pipeline using a small
deterministic hash-based fake embedder, so the test suite stays fast and
fully offline.

## Running the Benchmark

```bash
# Default: n=10000, dim=128, k=10, 200 queries
python benchmarks/benchmark_search.py

# Custom parameters
python benchmarks/benchmark_search.py --n 50000 --dim 256 --k 20 --queries 500
```

Distance computation (`l2_distance` / `cosine_distance`) is vectorized with
[xsimd](https://github.com/xtensor-stack/xsimd) (AVX2+FMA where the compiler
supports it, falling back to a portable width otherwise). Sample output
(n=10000, dim=128 — the benchmark's default):

```
Build time : 8.95s (1117 vec/s)
HNSW  ef=50  : 0.38ms / query, recall@10=0.782
HNSW  ef=200 : 1.11ms / query, recall@10=0.938
Brute-force  : 2.01ms / query (exact)
Speedup (ef=50 vs BF): 5.3x
```

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_client.py     # Phase 1 tests only
pytest tests/test_phase2.py     # Phase 2 tests only
pytest tests/test_phase2.py::TestHNSWRecall -v  # recall@k tests
pytest tests/test_rag_workflow.py  # Phase 3 RAG workflow tests (offline, no API keys needed)
```

## Project Structure

```
VectorHub/
├── vectorhub/
│   ├── __init__.py
│   ├── collection.py             # Python API (VectorCollection)
│   ├── rag.py                    # Phase 3: RAG helpers (chunking, embedding, generation)
│   └── _vectorhub.*.pyd          # compiled C++ extension
├── cpp/
│   ├── src/hnsw/
│   │   ├── hnsw.h                # HNSW class (true graph index)
│   │   └── hnsw.cpp              # HNSW + save/load + cosine + delete
│   └── pybind11_binding/
│       └── binding.cpp           # pybind11 bindings
├── tests/
│   ├── test_client.py            # Phase 1
│   ├── test_phase2.py            # Phase 2
│   └── test_rag_workflow.py      # Phase 3 (RAG workflow, offline)
├── benchmarks/
│   └── benchmark_search.py       # build/query perf + recall@k
├── examples/
│   ├── simple_demo.py
│   ├── rag_demo.py                # Phase 3: end-to-end RAG demo CLI
│   └── data/
│       └── sample.txt             # sample document for the RAG demo
├── CMakeLists.txt
└── pyproject.toml
```

## Current Limitations

- **Duplicate id re-insert after delete**: The HNSW graph keeps old edges from the deleted node; search still works correctly (lazy delete) but graph quality is slightly reduced. Full graph re-insertion is done when `update()` is called.
- **No cosine normalisation on insert**: Vectors are stored as-is; caller is responsible for normalising if using cosine metric.
- **Metadata filter is equality-only**: No range queries, no compound boolean logic, no indexing.
- **In-memory only**: The entire index lives in RAM; there is no disk-based paging.
- **Single-threaded**: No concurrent inserts or queries.
- **Windows runtime DLLs**: The MSYS2 build requires `libgcc_s_seh-1.dll`, `libstdc++-6.dll`, and `libwinpthread-1.dll` alongside the `.pyd` file.
- **RAG demo is single-document, char-based chunking**: `chunk_text()` splits on a fixed character window, not sentence/token boundaries, and the demo indexes one file per run (though `index_chunks()` can be called repeatedly with an `id_offset` to combine multiple sources, as shown in the tests).
- **No `batch_search()` yet**: the design doc mentions it, but batched multi-query search hasn't been implemented — the RAG demo only needs single-query search, done via the existing `search()`/`search(..., filter=...)`.
- **`openai` mode is an interface, not a validated integration**: `openai_generate_answer()` is implemented against the current `openai` Python SDK shape but has not been exercised against a live API key in this environment; `mock` mode is the default and is what's covered by tests.

## License

MIT
