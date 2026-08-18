"""RAG (Retrieval-Augmented Generation) helpers — Phase 3.

This module is a thin layer *on top of* the existing ``VectorCollection``
API (``add``, ``search``, ``save``/``load``). It does not touch the C++
core or the Phase 1/2 Python API — it only adds the plumbing needed to go
from raw text -> chunks -> embeddings -> index -> retrieval -> answer.

Everything here is optional: importing ``vectorhub`` does not require
``sentence-transformers`` (or any embedding backend) to be installed.
The dependency is only pulled in when :func:`get_default_embedder` (or the
demo script) is actually invoked.
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

EmbedFn = Callable[[Sequence[str]], List[List[float]]]


# ─── data loading ──────────────────────────────────────────────────────────

def load_text_file(path: str, encoding: str = "utf-8") -> str:
    """Load raw text from a local ``.txt`` file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    """Load raw text from a PDF file.

    PDF support is an *optional* capability — it depends on the ``pypdf``
    package, which is not a hard dependency of VectorHub. Install it with
    ``pip install vectorhub[pdf]`` (or ``pip install pypdf``) to use this
    function.

    Raises
    ------
    ImportError
        If ``pypdf`` is not installed.
    FileNotFoundError
        If ``path`` does not exist.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "PDF loading requires the optional 'pypdf' dependency. "
            "Install it with: pip install pypdf  (or: pip install vectorhub[pdf])"
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path: str) -> str:
    """Load a document's text, dispatching on file extension.

    ``.pdf`` -> :func:`load_pdf_file` (optional dependency)
    anything else -> :func:`load_text_file`
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_file(path)
    return load_text_file(path)


# ─── chunking ────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = 256,
    overlap: int = 32,
) -> List[Dict[str, Any]]:
    """Split ``text`` into overlapping, whitespace-normalized chunks.

    Parameters
    ----------
    text : str
        Raw input text.
    source : str
        Value stored in each chunk's ``source`` metadata field (typically
        the file name or path the text came from).
    chunk_size : int
        Chunk length in characters. Must be positive.
    overlap : int
        Number of characters shared between consecutive chunks. Must be
        ``0 <= overlap < chunk_size``.

    Returns
    -------
    list[dict]
        Each dict has keys ``chunk_id`` (int, 0-based index within this
        call), ``source`` (str), and ``text`` (str, non-empty).
        Empty/whitespace-only chunks are dropped, so the returned list may
        be shorter than a naive ``len(text) / step`` estimate.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    # Collapse all whitespace (including newlines) to single spaces so chunk
    # boundaries don't fall in the middle of ragged PDF/whitespace artifacts.
    normalized = re.sub(r"\s+", " ", text).strip()

    chunks: List[Dict[str, Any]] = []
    if not normalized:
        return chunks

    step = chunk_size - overlap
    pos = 0
    chunk_id = 0
    n = len(normalized)
    while pos < n:
        piece = normalized[pos:pos + chunk_size].strip()
        if piece:
            chunks.append({"chunk_id": chunk_id, "source": source, "text": piece})
            chunk_id += 1
        pos += step

    return chunks


# ─── embedding ───────────────────────────────────────────────────────────

def get_default_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbedFn:
    """Build an embedding function backed by ``sentence-transformers``.

    Returns
    -------
    callable
        ``embed(texts: list[str]) -> list[list[float]]``

    Raises
    ------
    ImportError
        If the ``sentence-transformers`` package is not installed.
    RuntimeError
        If the model itself fails to load (e.g. no network access to
        download it, or a corrupted local cache). The error message
        includes actionable next steps.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Embeddings require the optional 'sentence-transformers' "
            "dependency. Install it with: pip install sentence-transformers "
            "(or: pip install vectorhub[rag])"
        ) from e

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. This is usually "
            "caused by missing internet access (the model is downloaded from "
            "the Hugging Face Hub on first use) or a corrupted local cache.\n"
            "Things to try:\n"
            "  1. Check your internet connection and retry\n"
            "  2. Pre-download the model on a machine with network access, "
            "then copy the cache directory over\n"
            "  3. If the model is already cached locally, set the "
            "HF_HUB_OFFLINE=1 environment variable to force offline mode\n"
            f"Original error: {e}"
        ) from e

    def embed(texts: Sequence[str]) -> List[List[float]]:
        return model.encode(list(texts)).tolist()

    return embed


# ─── prompt & generation ─────────────────────────────────────────────────

def build_prompt(query: str, chunks: Sequence[Dict[str, Any]]) -> str:
    """Assemble a simple RAG prompt from retrieved chunks + the user query."""
    context = "\n\n".join(
        f"[{c.get('source', '?')}#{c.get('chunk_id', '?')}] {c.get('text', '')}"
        for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def mock_generate_answer(query: str, chunks: Sequence[Dict[str, Any]]) -> str:
    """Offline, dependency-free stand-in for an LLM call.

    Does not call any external API — it deterministically formats the
    retrieved chunks into a readable answer so the end-to-end RAG pipeline
    can run (and be tested) without network access or an API key.
    """
    if not chunks:
        return f"[mock] No relevant context was retrieved for: {query!r}"

    lines = [f"[mock answer] Based on {len(chunks)} retrieved chunk(s) for: {query!r}"]
    for c in chunks:
        snippet = c.get("text", "")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(
            f"- (source={c.get('source', '?')}, chunk_id={c.get('chunk_id', '?')}) {snippet}"
        )
    return "\n".join(lines)


def openai_generate_answer(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
) -> str:
    """Generate an answer via the OpenAI chat completion API.

    This is kept as a thin, optional integration point: VectorHub does not
    bundle a hard dependency on the ``openai`` package, and this function is
    never called by default (see :func:`generate_answer`'s ``mode="mock"``
    default). Callers must supply an API key explicitly or via the
    ``OPENAI_API_KEY`` environment variable.

    Raises
    ------
    RuntimeError
        If no API key is available.
    ImportError
        If the ``openai`` package is not installed.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI mode requires an API key. Set the OPENAI_API_KEY "
            "environment variable or pass api_key= explicitly. "
            "Use --llm mock to run the demo without any external API."
        )

    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "OpenAI mode requires the optional 'openai' package. "
            "Install it with: pip install openai"
        ) from e

    prompt = build_prompt(query, chunks)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def deepseek_generate_answer(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    model: str = "deepseek-v4-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Generate an answer via DeepSeek, through the LangChain framework.

    This uses ``langchain_deepseek.ChatDeepSeek`` (a thin LangChain chat
    model wrapper around DeepSeek's OpenAI-compatible API) instead of
    calling a raw HTTP/SDK client directly, so the same LangChain message
    plumbing (``build_prompt`` -> ``HumanMessage`` -> ``.invoke()``) can be
    reused/extended (e.g. streaming, callbacks, chains) later.

    VectorHub does not bundle a hard dependency on ``langchain`` /
    ``langchain-deepseek``: this function is only invoked when
    ``mode="deepseek"`` is explicitly requested. Callers must supply an API
    key explicitly or via the ``DEEPSEEK_API_KEY`` environment variable.

    Raises
    ------
    RuntimeError
        If no API key is available.
    ImportError
        If the ``langchain-deepseek`` package is not installed.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DeepSeek mode requires an API key. Set the DEEPSEEK_API_KEY "
            "environment variable or pass api_key= explicitly. "
            "Use --llm mock to run the demo without any external API."
        )

    try:
        from langchain_deepseek import ChatDeepSeek
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "DeepSeek mode requires the optional 'langchain-deepseek' "
            "(and 'langchain-core') packages. Install them with: "
            "pip install langchain-deepseek"
        ) from e

    prompt = build_prompt(query, chunks)
    llm = ChatDeepSeek(model=model, api_key=api_key, temperature=temperature)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_answer(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    mode: str = "mock",
    **kwargs,
) -> str:
    """Dispatch to the requested answer-generation backend.

    Parameters
    ----------
    mode : str
        ``"mock"`` (default, no external dependency), ``"openai"``, or
        ``"deepseek"`` (via LangChain).
    **kwargs
        Forwarded to :func:`openai_generate_answer` /
        :func:`deepseek_generate_answer` (e.g. ``model=``, ``api_key=``).
    """
    if mode == "mock":
        return mock_generate_answer(query, chunks)
    if mode == "openai":
        return openai_generate_answer(query, chunks, **kwargs)
    if mode == "deepseek":
        return deepseek_generate_answer(query, chunks, **kwargs)
    raise ValueError(f"Unknown llm mode: {mode!r}. Use 'mock', 'openai', or 'deepseek'.")


# ─── indexing / retrieval convenience wrappers ──────────────────────────
#
# These simply compose the existing VectorCollection.add()/search() API —
# no new index capability is introduced.

def index_chunks(collection, chunks: Sequence[Dict[str, Any]], embed_fn: EmbedFn,
                  id_offset: int = 0) -> List[int]:
    """Embed ``chunks`` and add them to an existing ``VectorCollection``.

    Each chunk's full dict (``source``, ``chunk_id``, ``text``) is stored as
    that vector's metadata, so search results can report back the original
    text without a separate lookup table.

    Returns
    -------
    list[int]
        The ids assigned to the inserted chunks (``id_offset .. id_offset+n-1``).
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_fn(texts)
    ids = list(range(id_offset, id_offset + len(chunks)))
    metadatas = [dict(c) for c in chunks]
    collection.add(vectors, ids, metadatas=metadatas)
    return ids


def retrieve(collection, query: str, embed_fn: EmbedFn, k: int = 5,
             filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Embed ``query`` and search ``collection``, returning enriched chunk dicts.

    Returns
    -------
    list[dict]
        Each result dict is the stored chunk metadata plus ``id`` and
        ``distance`` keys, ordered by increasing distance.
    """
    query_vector = embed_fn([query])[0]
    raw = collection.search(query_vector, k=k, filter=filter, include_metadata=True)
    results = []
    for id_, dist, meta in raw:
        entry = dict(meta)
        entry["id"] = id_
        entry["distance"] = dist
        results.append(entry)
    return results
