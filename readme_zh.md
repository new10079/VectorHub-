# VectorHub — 轻量级向量数据库

一个轻量级的 Python + C++ 混合向量数据库，支持 HNSW 索引。
**阶段二**增加了真正的 HNSW 图搜索、余弦距离、持久化、元数据、删除/更新以及基准测试工具。**阶段三**在已有的阶段一/二 API 之上，完整实现了一个端到端的 RAG（检索增强生成）演示。

> **定位**：比 Milvus 更轻量，比 Faiss 更易用。

## 功能特性

|                                    | 阶段一   | 阶段二          | 阶段三（当前）              |
| ---------------------------------- | -------- | --------------- | --------------------------- |
| C++ HNSW 索引                      | 暴力搜索 | 真正的 HNSW 图  | —                           |
| 距离度量                           | 仅 L2    | **L2 + 余弦**   | —                           |
| HNSW 参数（M、ef）                 | —        | ✓               | —                           |
| 持久化（保存/加载）                | —        | ✓               | ✓（RAG 演示使用）           |
| 元数据存储                         | —        | ✓               | ✓（用于存储来源/分块/文本） |
| 元数据过滤                         | —        | ✓（等值匹配）   | ✓（用于按来源检索）         |
| 删除/更新                          | —        | ✓               | —                           |
| 暴力搜索                           | —        | ✓（召回率测试） | —                           |
| 基准测试脚本                       | —        | ✓               | —                           |
| 文本加载/分块                      | —        | —               | ✓（`vectorhub.rag`）        |
| RAG 演示（`examples/rag_demo.py`） | —        | —               | ✓                           |

## 安装

### 1. 创建虚拟环境（推荐）

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

> 虚拟环境可以把本项目的依赖与系统 Python 隔离，是使用 VectorHub 的推荐方式。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

这会安装构建工具链（CMake、Ninja、pybind11、scikit-build-core）、核心运行时
（NumPy）和 pytest。RAG 演示的可选依赖可随后安装：

```bash
pip install -e ".[rag,pdf,openai,deepseek,dotenv]"
```

> **中国大陆网络提示**：部分网络环境下 PyPI 和 GitHub 可能无法访问。如果
> `pip install` 超时，请使用镜像源，例如：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. 安装项目（自动构建 C++ 扩展）

```bash
pip install -e .
```

`scikit-build-core` 会自动调用 CMake 构建，因此需要确保 `cmake`、`ninja` 和
C++17 编译器（Windows 上用 g++ 7+ / MSYS2 ucrt64，Linux/macOS 上用 GCC/Clang）
已在 `PATH` 中。首次配置时会自动下载
[xsimd](https://github.com/xtensor-stack/xsimd)（仅头文件的 SIMD 库，通过 CMake
`FetchContent` 拉取，之后会缓存复用）。

### 4. 验证安装

```bash
python -c "from vectorhub import VectorCollection; print('OK')"
python examples/simple_demo.py
pytest
```

### 手动构建（可选）

如果不想让 `pip install -e .` 自动构建，也可以手动编译 C++ 扩展：

```bash
pip install cmake pybind11 ninja scikit-build-core numpy pytest

mkdir build_cpp && cd build_cpp
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build . --config Release
cd ..

# 将编译好的扩展复制到包中
cp build_cpp/_vectorhub*.pyd vectorhub/

# Windows 上使用 MSYS2 时，还需复制运行时 DLL：
# cp D:/msys2/ucrt64/bin/libgcc_s_seh-1.dll vectorhub/
# cp D:/msys2/ucrt64/bin/libwinpthread-1.dll vectorhub/
# cp D:/msys2/ucrt64/bin/libstdc++-6.dll     vectorhub/
```

## 快速开始

```python
from vectorhub import VectorCollection

db = VectorCollection(dim=128, metric="l2", M=16, ef_construction=200, ef_search=50)

db.add(vectors=my_vectors, ids=my_ids)

results = db.search(query_vector, k=10)
for id_, dist in results:
    print(id_, dist)
```

## API 参考

### `VectorCollection(dim, metric="l2", M=16, ef_construction=200, ef_search=50)`

| 参数              | 默认值 | 含义                                                         |
| ----------------- | ------ | ------------------------------------------------------------ |
| `dim`             | 必填   | 向量维度                                                     |
| `metric`          | `"l2"` | `"l2"`（欧氏距离）或 `"cosine"`（余弦距离）                  |
| `M`               | 16     | HNSW 每层每个节点的最大邻居数。值越高 → 召回率越好，内存占用越大 |
| `ef_construction` | 200    | 构建期间的候选池大小。值越高 → 图质量越好，插入越慢          |
| `ef_search`       | 50     | 搜索期间的候选池大小。值越高 → 召回率越好，查询越慢          |

### `add(vectors, ids, metadatas=None)`

添加向量。`metadatas` 是可选的长度相等的 `list[dict]`。

```python
db.add(
    vectors=[[0.1, 0.2, ...], ...],
    ids=[1, 2, 3],
    metadatas=[{"source": "wiki", "year": 2024}, {"source": "news"}, None],
)
```

重复 ID 会抛出 `RuntimeError`。维度不匹配会抛出 `ValueError`。

### `search(query, k=10, filter=None, include_metadata=False)`

默认返回 `list[(id, distance)]`。

```python
# 基础搜索
results = db.search(query, k=5)

# 返回元数据
results = db.search(query, k=5, include_metadata=True)
# → [(id, distance, {"source": "wiki"}), ...]

# 元数据等值过滤
results = db.search(query, k=5, filter={"source": "wiki"})
```

### `brute_force_search(query, k=10)`

精确线性扫描搜索。用于召回率基准测试或生成 ground-truth 数据。

### `delete(id)`

惰性删除向量。已删除的 ID 将从后续搜索中排除。

```python
db.delete(42)
```

如果 ID 不存在或已被删除，抛出 `KeyError`。

### `update(id, vector, metadata=None)`

删除 + 重新插入。图边会被更新。

```python
db.update(42, new_vector, metadata={"version": 2})
```

### `save(path)` / `VectorCollection.load(path)`

持久化到两个文件：`<path>`（二进制索引）和 `<path>.meta.json`（元数据）。

```python
db.save("/tmp/my_index.vhb")

db2 = VectorCollection.load("/tmp/my_index.vhb")
# db2 立即可用于搜索和继续添加向量
```

### `set_ef_search(ef)`

运行时调整搜索质量，无需重建索引。

```python
db.set_ef_search(200)  # 更高召回率，更慢
db.set_ef_search(20)   # 更快，更低召回率
```

### `len(db)`

返回存活（未被删除）的向量数量。

## HNSW 参数指南

| 场景                 | M    | ef_construction | ef_search |
| -------------------- | ---- | --------------- | --------- |
| 快速原型（小数据集） | 8    | 50              | 20        |
| 默认（均衡）         | 16   | 200             | 50        |
| 高召回率（生产环境） | 32   | 400             | 100       |
| 最高召回率           | 48   | 500             | 200+      |

**经验法则**：优先增加 `ef_search`（成本最低），如果召回率仍不满足要求再增加 `M`。

## RAG 演示（阶段三）

`examples/rag_demo.py` 演示了一个完整的检索增强生成管道，完全基于现有的 `VectorCollection` API 构建——无需修改 C++ 核心代码。

管道流程：**加载**文档 → **分块** → **嵌入**每个分块 → **索引**到 `VectorCollection`（附带 `source` / `chunk_id` / `text` 元数据）→ 针对问题**检索** top-k 分块 → **生成**答案。

### 安装额外依赖

嵌入使用 [sentence-transformers](https://www.sbert.net/) 生成，这是一个*可选*依赖（不是使用 VectorHub 核心所必需的）：

```bash
pip install sentence-transformers
# 或：pip install vectorhub[rag]
```

可选的 PDF 加载支持：

```bash
pip install pypdf
# 或：pip install vectorhub[pdf]
```

如果模型无法下载（无网络/缓存损坏），`vectorhub.rag.get_default_embedder()` 和演示脚本都会抛出明确的错误，说明解决方法（检查网络、预先下载模型，或如果已缓存则设置 `HF_HUB_OFFLINE=1`）。

> 如果从您的网络无法访问 `huggingface.co`（在中国大陆较常见），请在运行演示前设置 `HF_ENDPOINT=https://hf-mirror.com`——本仓库正是使用此镜像站完成端到端验证的。

### 运行演示

```bash
python examples/rag_demo.py \
    --data examples/data/sample.txt \
    --query "What is VectorHub?" \
    --top-k 3 \
    --llm mock
```

使用持久化（复用阶段二的 `save()`/`load()` API——索引只构建一次，后续运行从磁盘重新加载，无需重新嵌入）：

```bash
python examples/rag_demo.py \
    --data examples/data/sample.txt \
    --query "What is VectorHub?" \
    --top-k 3 \
    --llm mock \
    --persist-path examples/data/vectorhub_index.bin
```

### CLI 选项

| 参数                | 默认值             | 含义                                                         |
| ------------------- | ------------------ | ------------------------------------------------------------ |
| `--data`            | 必填               | `.txt` 文件路径（安装 `pypdf` 后支持 `.pdf`）                |
| `--query`           | 必填               | 要提问的问题                                                 |
| `--top-k`           | 3                  | 检索的分块数量                                               |
| `--chunk-size`      | 256                | 分块大小（字符数）                                           |
| `--overlap`         | 32                 | 连续分块之间的重叠（字符数）                                 |
| `--llm`             | `mock`             | `mock`（离线，无依赖）、`openai` 或 `deepseek`（通过 LangChain） |
| `--embedding-model` | `all-MiniLM-L6-v2` | sentence-transformers 模型名称                               |
| `--persist-path`    | 无                 | 如果设置，将索引保存到该路径，下次运行时重新加载             |

### `mock` vs `openai` vs `deepseek` 答案生成

- **`mock`（默认）**——不调用外部 API。将检索到的分块格式化为可读的答案摘要。这使得演示可以在零 API 密钥、除一次性嵌入模型下载外零网络访问的情况下运行。
- **`openai`**——调用 OpenAI 聊天补全 API。需要可选的 `openai` 包（`pip install openai` / `pip install vectorhub[openai]`）和 API 密钥，可通过向 `vectorhub.rag.openai_generate_answer()` 传入 `api_key=` 参数或设置 `OPENAI_API_KEY` 环境变量提供。如果未设置密钥，会抛出 `RuntimeError` 提示使用 `--llm mock`——绝不会静默降级。
- **`deepseek`**——通过 LangChain 框架（`langchain_deepseek.ChatDeepSeek`）调用 DeepSeek 聊天 API。需要可选的 `langchain-deepseek` 包（`pip install langchain-deepseek` / `pip install vectorhub[deepseek]`）和 API 密钥，可通过向 `vectorhub.rag.deepseek_generate_answer()` 传入 `api_key=` 参数或设置 `DEEPSEEK_API_KEY` 环境变量提供。同样不会静默降级。

`examples/rag_demo.py` 还会自动加载仓库根目录下的 `.env` 文件（通过可选的 `python-dotenv` 包——`pip install python-dotenv` / `pip install vectorhub[dotenv]`），因此在此文件中设置的 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 会被自动读取，无需在 shell 中手动导出。如果未安装 `python-dotenv`，此步骤会被静默跳过，通过其他方式设置的环境变量仍然有效。

### `vectorhub.rag` 模块

可复用的构建模块（每个都可独立测试，均不触碰 C++ 核心）：

```python
from vectorhub.rag import (
    load_document,       # .txt（始终支持）/ .pdf（需要 pypdf）
    chunk_text,           # 文本 -> list[{chunk_id, source, text}]
    get_default_embedder, # -> callable(list[str]) -> list[list[float]]
    index_chunks,         # 分块 + 嵌入函数 -> 插入到 VectorCollection
    retrieve,             # 查询 + 嵌入函数 -> 排序后的分块字典列表
    build_prompt,
    mock_generate_answer,
    openai_generate_answer,
    deepseek_generate_answer,
    generate_answer,       # 根据 mode="mock" | "openai" | "deepseek" 分发
)
```

`chunk_text` 和 `mock_generate_answer` 没有任何依赖；`get_default_embedder` 需要 `sentence-transformers`；`openai_generate_answer` 需要 `openai` + API 密钥；`deepseek_generate_answer` 需要 `langchain-deepseek`（+ `langchain-core`）和 `DEEPSEEK_API_KEY`。`tests/test_rag_workflow.py` 使用一个小型确定性哈希伪嵌入器来测试整个管道，因此测试套件保持快速且完全离线。

## 运行基准测试

```bash
# 默认：n=10000，dim=128，k=10，200 个查询
python benchmarks/benchmark_search.py

# 自定义参数
python benchmarks/benchmark_search.py --n 50000 --dim 256 --k 20 --queries 500
```

距离计算（`l2_distance` / `cosine_distance`）使用 [xsimd](https://github.com/xtensor-stack/xsimd) 进行向量化（在编译器支持的情况下使用 AVX2+FMA，否则回退到可移植的宽度）。示例输出（n=10000，dim=128——基准测试的默认配置）：

```
Build time : 8.95s (1117 vec/s)
HNSW  ef=50  : 0.38ms / query, recall@10=0.782
HNSW  ef=200 : 1.11ms / query, recall@10=0.938
Brute-force  : 2.01ms / query (exact)
Speedup (ef=50 vs BF): 5.3x
```

## 运行测试

```bash
pytest                          # 所有测试
pytest tests/test_client.py     # 仅阶段一测试
pytest tests/test_phase2.py     # 仅阶段二测试
pytest tests/test_phase2.py::TestHNSWRecall -v  # recall@k 测试
pytest tests/test_rag_workflow.py  # 阶段三 RAG 工作流测试（离线，无需 API 密钥）
```

## 项目结构

```
VectorHub/
├── vectorhub/
│   ├── __init__.py
│   ├── collection.py             # Python API（VectorCollection）
│   ├── rag.py                    # 阶段三：RAG 辅助函数（分块、嵌入、生成）
│   └── _vectorhub.*.pyd          # 编译后的 C++ 扩展
├── cpp/
│   ├── src/hnsw/
│   │   ├── hnsw.h                # HNSW 类（真正的图索引）
│   │   └── hnsw.cpp              # HNSW + 保存/加载 + 余弦 + 删除
│   └── pybind11_binding/
│       └── binding.cpp           # pybind11 绑定
├── tests/
│   ├── test_client.py            # 阶段一
│   ├── test_phase2.py            # 阶段二
│   └── test_rag_workflow.py      # 阶段三（RAG 工作流，离线）
├── benchmarks/
│   └── benchmark_search.py       # 构建/查询性能 + recall@k
├── examples/
│   ├── simple_demo.py
│   ├── rag_demo.py                # 阶段三：端到端 RAG 演示 CLI
│   └── data/
│       └── sample.txt             # RAG 演示使用的示例文档
├── CMakeLists.txt
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## 当前限制

- **删除后重复插入相同 ID**：HNSW 图会保留已删除节点的旧边；搜索仍能正常工作（惰性删除），但图质量略有下降。调用 `update()` 时会完整重新插入。
- **插入时不做余弦归一化**：向量按原样存储；使用余弦度量时，调用方负责自行归一化。
- **元数据过滤仅支持等值匹配**：不支持范围查询、复合布尔逻辑或索引。
- **纯内存运行**：整个索引驻留在 RAM 中，不支持基于磁盘的分页。
- **单线程**：不支持并发插入或查询。
- **Windows 运行时 DLL**：MSYS2 构建需要将 `libgcc_s_seh-1.dll`、`libstdc++-6.dll` 和 `libwinpthread-1.dll` 与 `.pyd` 文件放在一起。
- **RAG 演示为单文档、基于字符的分块**：`chunk_text()` 按固定字符窗口分割，而不是按句子/标记边界；演示每次运行索引一个文件（尽管 `index_chunks()` 可以通过 `id_offset` 参数重复调用来合并多个来源，如测试中所示）。
- **尚无 `batch_search()`**：设计文档中提到了批处理搜索，但尚未实现多查询批处理——RAG 演示只需单查询搜索，通过现有的 `search()`/`search(..., filter=...)` 即可满足。
- **`openai` 模式是接口而非经过验证的集成**：`openai_generate_answer()` 按照当前 `openai` Python SDK 的形式实现，但尚未在本环境中用真实的 API 密钥进行测试；`mock` 模式是默认模式，也是测试所覆盖的模式。

## 许可证

MIT