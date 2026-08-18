#include "hnsw.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <xsimd/xsimd.hpp>

// ─── Constructor ─────────────────────────────────────────────────────────────

HNSW::HNSW(int dim, const std::string& metric, int M, int ef_construction, int ef_search)
    : dim_(dim), metric_(metric), M_(M),
      ef_construction_(ef_construction), ef_search_(ef_search),
      entry_point_(-1), cur_max_level_(-1),
      rng_(42)
{
    if (dim <= 0)
        throw std::invalid_argument("Dimension must be positive");
    if (metric != "l2" && metric != "cosine")
        throw std::invalid_argument("Unsupported metric: " + metric + ". Use 'l2' or 'cosine'");
    if (M < 2)
        throw std::invalid_argument("M must be >= 2");
    if (ef_construction < M)
        throw std::invalid_argument("ef_construction must be >= M");
    if (ef_search < 1)
        throw std::invalid_argument("ef_search must be >= 1");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

float HNSW::l2_distance(const std::vector<float>& a, const std::vector<float>& b) const {
    using batch = xsimd::batch<float>;
    constexpr std::size_t simd_size = batch::size;
    std::size_t n = static_cast<std::size_t>(dim_);
    std::size_t vec_n = n - n % simd_size;

    batch sum_vec(0.0f);
    for (std::size_t i = 0; i < vec_n; i += simd_size) {
        batch av = batch::load_unaligned(a.data() + i);
        batch bv = batch::load_unaligned(b.data() + i);
        batch d = av - bv;
        sum_vec += d * d;
    }
    float sum = xsimd::reduce_add(sum_vec);
    for (std::size_t i = vec_n; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

float HNSW::cosine_distance(const std::vector<float>& a, const std::vector<float>& b) const {
    using batch = xsimd::batch<float>;
    constexpr std::size_t simd_size = batch::size;
    std::size_t n = static_cast<std::size_t>(dim_);
    std::size_t vec_n = n - n % simd_size;

    batch dot_vec(0.0f), na_vec(0.0f), nb_vec(0.0f);
    for (std::size_t i = 0; i < vec_n; i += simd_size) {
        batch av = batch::load_unaligned(a.data() + i);
        batch bv = batch::load_unaligned(b.data() + i);
        dot_vec += av * bv;
        na_vec  += av * av;
        nb_vec  += bv * bv;
    }
    float dot = xsimd::reduce_add(dot_vec);
    float na  = xsimd::reduce_add(na_vec);
    float nb  = xsimd::reduce_add(nb_vec);
    for (std::size_t i = vec_n; i < n; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-10f) return 1.0f;
    float sim = dot / denom;
    if (sim > 1.0f) sim = 1.0f;
    if (sim < -1.0f) sim = -1.0f;
    return 1.0f - sim;
}

float HNSW::distance(const std::vector<float>& a, const std::vector<float>& b) const {
    if (metric_ == "l2")     return l2_distance(a, b);
    if (metric_ == "cosine") return cosine_distance(a, b);
    throw std::runtime_error("Unknown metric: " + metric_);
}

float HNSW::dot_product(const std::vector<float>& a, const std::vector<float>& b) const {
    using batch = xsimd::batch<float>;
    constexpr std::size_t simd_size = batch::size;
    std::size_t n = static_cast<std::size_t>(dim_);
    std::size_t vec_n = n - n % simd_size;

    batch dot_vec(0.0f);
    for (std::size_t i = 0; i < vec_n; i += simd_size) {
        batch av = batch::load_unaligned(a.data() + i);
        batch bv = batch::load_unaligned(b.data() + i);
        dot_vec += av * bv;
    }
    float dot = xsimd::reduce_add(dot_vec);
    for (std::size_t i = vec_n; i < n; ++i)
        dot += a[i] * b[i];
    return dot;
}

float HNSW::node_distance(int i, int j) const {
    if (metric_ != "cosine")
        return distance(node_vectors_[i], node_vectors_[j]);

    // Same formula as cosine_distance(), but na/nb come from the cached
    // per-node norms instead of being recomputed from the vectors.
    float dot = dot_product(node_vectors_[i], node_vectors_[j]);
    float denom = std::sqrt(node_norms_[i]) * std::sqrt(node_norms_[j]);
    if (denom < 1e-10f) return 1.0f;
    float sim = dot / denom;
    if (sim > 1.0f) sim = 1.0f;
    if (sim < -1.0f) sim = -1.0f;
    return 1.0f - sim;
}

int HNSW::random_level() const {
    // Standard HNSW level assignment: geometric distribution with mL = 1/ln(M)
    double mL = 1.0 / std::log(static_cast<double>(M_));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int level = 0;
    while (dist(rng_) < std::exp(-1.0 / mL) && level < 32)
        ++level;
    return level;
}

int HNSW::size() const {
    return static_cast<int>(node_ids_.size()) - static_cast<int>(deleted_.size());
}

// ─── Insert ──────────────────────────────────────────────────────────────────

void HNSW::insert(int id, const std::vector<float>& vector) {
    if (static_cast<int>(vector.size()) != dim_)
        throw std::invalid_argument("Vector dimension mismatch: expected " +
                                    std::to_string(dim_) + ", got " +
                                    std::to_string(vector.size()));
    if (id_to_idx_.count(id) > 0) {
        // Check if it's a "live" id (not deleted)
        if (!deleted_.count(id))
            throw std::runtime_error("Duplicate id: " + std::to_string(id));

        // Re-inserting a previously-deleted id: treat it exactly like a brand
        // new node. First strip every edge it held under its old vector
        // position (both its own neighbor lists and the reverse references
        // its old neighbors hold), then re-run the normal greedy insertion
        // procedure with a freshly-sampled level. This avoids leaving stale
        // connections in the graph that no longer reflect where the vector
        // actually lives.
        int idx = id_to_idx_[id];
        disconnect_node(idx, node_levels_[idx]);

        deleted_.erase(id);
        node_vectors_[idx] = vector;
        if (metric_ == "cosine")
            node_norms_[idx] = dot_product(vector, vector);

        int node_level = random_level();
        node_levels_[idx] = node_level;
        connect_new_node(idx, vector, node_level);
        return;
    }

    int new_idx = static_cast<int>(node_ids_.size());
    node_ids_.push_back(id);
    node_vectors_.push_back(vector);
    node_norms_.push_back(metric_ == "cosine" ? dot_product(vector, vector) : 0.0f);
    node_levels_.push_back(0); // overwritten below
    id_to_idx_[id] = new_idx;

    int node_level = random_level();
    node_levels_[new_idx] = node_level;

    if (entry_point_ == -1) {
        // First node in the index
        while (static_cast<int>(layers_.size()) <= node_level)
            layers_.push_back(std::vector<std::vector<int>>());
        for (int lv = 0; lv <= node_level; ++lv)
            while (static_cast<int>(layers_[lv].size()) <= new_idx)
                layers_[lv].push_back(std::vector<int>());

        entry_point_ = new_idx;
        cur_max_level_ = node_level;
        return;
    }

    connect_new_node(new_idx, vector, node_level);
}

void HNSW::insert_batch(const std::vector<int>& ids,
                         const std::vector<std::vector<float>>& vectors) {
    if (ids.size() != vectors.size())
        throw std::invalid_argument("Number of ids and vectors must match");
    for (size_t i = 0; i < ids.size(); ++i)
        insert(ids[i], vectors[i]);
}

// ─── Connect / disconnect helpers ────────────────────────────────────────────

// Links node `idx` into the graph at levels [0, node_level], as if it were
// being freshly inserted. Shared by fresh inserts and by re-inserts of a
// previously-deleted id (after disconnect_node() has cleared its old edges).
void HNSW::connect_new_node(int idx, const std::vector<float>& vector, int node_level) {
    // Ensure layers_ has enough levels
    while (static_cast<int>(layers_.size()) <= node_level)
        layers_.push_back(std::vector<std::vector<int>>());
    // Ensure every level up to node_level has an entry for idx
    for (int lv = 0; lv <= node_level; ++lv)
        while (static_cast<int>(layers_[lv].size()) <= idx)
            layers_[lv].push_back(std::vector<int>());

    // Search from top level down to node_level+1 to find the best entry point
    int ep = entry_point_;
    for (int lv = cur_max_level_; lv > node_level; --lv) {
        auto candidates = search_layer(vector, ep, 1, lv);
        if (!candidates.empty())
            ep = candidates.top().second;
    }

    // Insert at each level from min(node_level, cur_max_level) down to 0
    for (int lv = std::min(node_level, cur_max_level_); lv >= 0; --lv) {
        int M_lv = (lv == 0) ? (2 * M_) : M_;

        auto candidates = search_layer(vector, ep, ef_construction_, lv);

        // Update ep to best found
        if (!candidates.empty())
            ep = candidates.top().second;

        auto neighbors = select_neighbors(vector, candidates, M_lv);

        // Connect node -> neighbors
        layers_[lv][idx] = neighbors;

        // Connect neighbors -> node (bidirectional), prune if over M_lv
        for (int nb : neighbors) {
            auto& nb_conns = layers_[lv][nb];
            if (std::find(nb_conns.begin(), nb_conns.end(), idx) == nb_conns.end())
                nb_conns.push_back(idx);
            if (static_cast<int>(nb_conns.size()) > M_lv) {
                // Prune: keep M_lv closest. Precompute each candidate's
                // distance once instead of recomputing it on every sort
                // comparison (the comparator previously called distance()
                // twice per comparison, i.e. O(n log n) distance calls).
                std::vector<std::pair<float,int>> dist_idx;
                dist_idx.reserve(nb_conns.size());
                for (int c : nb_conns)
                    dist_idx.push_back({node_distance(nb, c), c});
                std::sort(dist_idx.begin(), dist_idx.end());

                nb_conns.clear();
                for (int i = 0; i < M_lv; ++i)
                    nb_conns.push_back(dist_idx[i].second);
            }
        }
    }

    if (node_level > cur_max_level_) {
        cur_max_level_ = node_level;
        entry_point_ = idx;
    } else if (idx == entry_point_ && node_level < cur_max_level_) {
        // This node used to be the global entry point (it held the highest
        // level in the graph) but was just re-linked at a lower level — find
        // another live node that still reaches cur_max_level_, or shrink
        // cur_max_level_ down to whatever the new tallest node is.
        int best_idx = -1, best_level = -1;
        for (int i = 0; i < static_cast<int>(node_ids_.size()); ++i) {
            if (i == idx || deleted_.count(node_ids_[i])) continue;
            if (node_levels_[i] > best_level) { best_level = node_levels_[i]; best_idx = i; }
        }
        if (best_idx == -1) {
            entry_point_ = idx;
            cur_max_level_ = node_level;
        } else {
            entry_point_ = best_idx;
            cur_max_level_ = best_level;
        }
    }
}

// Strips node `idx`'s existing edges (its own neighbor lists and the reverse
// references its neighbors hold) at every level up to old_level.
void HNSW::disconnect_node(int idx, int old_level) {
    for (int lv = 0; lv <= old_level && lv < static_cast<int>(layers_.size()); ++lv) {
        if (idx >= static_cast<int>(layers_[lv].size())) continue;
        for (int nb : layers_[lv][idx]) {
            if (nb >= static_cast<int>(layers_[lv].size())) continue;
            auto& nb_conns = layers_[lv][nb];
            nb_conns.erase(std::remove(nb_conns.begin(), nb_conns.end(), idx), nb_conns.end());
        }
        layers_[lv][idx].clear();
    }
}


// ─── Search layer ────────────────────────────────────────────────────────────

// Returns max-heap of (dist, internal_idx) with at most ef elements
std::priority_queue<std::pair<float,int>>
HNSW::search_layer(const std::vector<float>& query,
                   int ep_idx,
                   int ef,
                   int level) const {
    // visited set
    std::unordered_set<int> visited;
    visited.insert(ep_idx);

    float ep_dist = distance(query, node_vectors_[ep_idx]);

    // candidates: min-heap (closest first) for expansion
    std::priority_queue<std::pair<float,int>,
                        std::vector<std::pair<float,int>>,
                        std::greater<std::pair<float,int>>> candidates;
    candidates.push({ep_dist, ep_idx});

    // result: max-heap (furthest first), bounded to ef
    std::priority_queue<std::pair<float,int>> result;
    result.push({ep_dist, ep_idx});

    while (!candidates.empty()) {
        auto [c_dist, c_idx] = candidates.top();
        candidates.pop();

        // If closest candidate is further than furthest result, stop
        if (!result.empty() && c_dist > result.top().first)
            break;

        // Expand neighbors of c_idx at this level
        if (level < static_cast<int>(layers_.size()) &&
            c_idx < static_cast<int>(layers_[level].size())) {
            for (int nb : layers_[level][c_idx]) {
                if (visited.count(nb)) continue;
                visited.insert(nb);

                float nb_dist = distance(query, node_vectors_[nb]);

                if (static_cast<int>(result.size()) < ef || nb_dist < result.top().first) {
                    candidates.push({nb_dist, nb});
                    result.push({nb_dist, nb});
                    if (static_cast<int>(result.size()) > ef)
                        result.pop(); // keep only ef closest
                }
            }
        }
    }
    return result;
}

// ─── Select neighbors ────────────────────────────────────────────────────────

std::vector<int>
HNSW::select_neighbors(const std::vector<float>& vec,
                        const std::priority_queue<std::pair<float,int>>& candidates,
                        int M_max) const {
    // Convert max-heap to sorted vector (closest first)
    std::vector<std::pair<float,int>> sorted;
    auto tmp = candidates;
    while (!tmp.empty()) {
        sorted.push_back(tmp.top());
        tmp.pop();
    }
    std::sort(sorted.begin(), sorted.end()); // ascending by dist

    // SELECT-NEIGHBORS-HEURISTIC (Malkov & Yashunin, Algorithm 4), with
    // extendCandidates=false and keepPrunedConnections=true.
    //
    // Plain "closest M" picks neighbors that are all near `vec` but says
    // nothing about how they relate to *each other* -- in dense regions
    // they end up clustered in the same direction, so a node's edges all
    // point one way instead of spanning the space around it. As the index
    // grows this starves the graph of the long-range edges greedy search
    // needs to jump between regions, so queries take more hops (latency
    // grows much faster than N) and are more likely to get stuck in a
    // local optimum before reaching the true top-k (recall drops even as
    // ef_search stays fixed).
    //
    // The heuristic instead keeps a candidate only if it is closer to
    // `vec` than to every neighbor already selected -- i.e. it adds
    // information not already covered by an existing edge -- which keeps
    // the selected set spread out around `vec` rather than bunched up.
    std::vector<int> result;
    std::vector<std::pair<float,int>> discarded;
    result.reserve(M_max);

    for (auto& [d_q, idx] : sorted) {
        if (static_cast<int>(result.size()) >= M_max) break;

        bool is_diverse = true;
        for (int r : result) {
            float d_to_selected = node_distance(idx, r);
            if (d_to_selected < d_q) {
                is_diverse = false;
                break;
            }
        }

        if (is_diverse)
            result.push_back(idx);
        else
            discarded.push_back({d_q, idx});
    }

    // keepPrunedConnections: if the diversity filter left the node under
    // M_max edges, top up with the closest discarded candidates rather
    // than leaving it under-connected.
    for (auto& [d, idx] : discarded) {
        if (static_cast<int>(result.size()) >= M_max) break;
        result.push_back(idx);
    }

    return result;
}

// ─── Public search ────────────────────────────────────────────────────────────

std::vector<std::pair<int, float>>
HNSW::search(const std::vector<float>& query, int k) const {
    if (static_cast<int>(query.size()) != dim_)
        throw std::invalid_argument("Query dimension mismatch: expected " +
                                    std::to_string(dim_) + ", got " +
                                    std::to_string(query.size()));
    if (k <= 0)
        throw std::invalid_argument("k must be positive");

    int live_count = size();
    if (live_count == 0) return {};

    if (entry_point_ == -1) return {};

    // Greedy descent from top level to level 1
    int ep = entry_point_;
    if (deleted_.count(node_ids_[ep])) {
        // entry_point_ itself has been lazily deleted — fall back to any
        // live node as a starting point (its vector value is still valid,
        // just not necessarily a great starting position).
        for (int i = 0; i < static_cast<int>(node_ids_.size()); ++i) {
            if (!deleted_.count(node_ids_[i])) { ep = i; break; }
        }
    }
    for (int lv = cur_max_level_; lv > 0; --lv) {
        auto candidates = search_layer(query, ep, 1, lv);
        if (!candidates.empty())
            ep = candidates.top().second;
    }

    // At level 0: search with ef_search
    int ef = std::max(ef_search_, k);
    auto result_heap = search_layer(query, ep, ef, 0);

    // Collect, filter deleted, sort
    std::vector<std::pair<float,int>> all;
    while (!result_heap.empty()) {
        all.push_back(result_heap.top());
        result_heap.pop();
    }
    std::sort(all.begin(), all.end()); // ascending dist

    std::vector<std::pair<int, float>> results;
    for (auto& [d, idx] : all) {
        int ext_id = node_ids_[idx];
        if (deleted_.count(ext_id)) continue;
        results.push_back({ext_id, d});
        if (static_cast<int>(results.size()) >= k) break;
    }
    return results;
}

// ─── Brute-force search ───────────────────────────────────────────────────────

std::vector<std::pair<int, float>>
HNSW::brute_force_search(const std::vector<float>& query, int k) const {
    if (static_cast<int>(query.size()) != dim_)
        throw std::invalid_argument("Query dimension mismatch");
    if (k <= 0)
        throw std::invalid_argument("k must be positive");

    std::vector<std::pair<int, float>> results;
    for (size_t i = 0; i < node_ids_.size(); ++i) {
        if (deleted_.count(node_ids_[i])) continue;
        float d = distance(query, node_vectors_[i]);
        results.push_back({node_ids_[i], d});
    }

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (k < static_cast<int>(results.size()))
        results.resize(k);
    return results;
}

// ─── Delete ───────────────────────────────────────────────────────────────────

void HNSW::remove(int id) {
    if (id_to_idx_.find(id) == id_to_idx_.end())
        throw std::out_of_range("ID not found: " + std::to_string(id));
    if (deleted_.count(id))
        throw std::out_of_range("ID already deleted: " + std::to_string(id));
    deleted_.insert(id);
}

// ─── Persistence ──────────────────────────────────────────────────────────────

void HNSW::save(const std::string& filepath) const {
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file for writing: " + filepath);

    auto write_int = [&](int v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto write_float = [&](float v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto write_str = [&](const std::string& s) {
        int len = static_cast<int>(s.size());
        write_int(len);
        f.write(s.data(), len);
    };

    // Header
    write_int(FILE_VERSION);
    write_int(dim_);
    write_str(metric_);
    write_int(M_);
    write_int(ef_construction_);
    write_int(ef_search_);
    write_int(entry_point_);
    write_int(cur_max_level_);

    // Nodes
    int n = static_cast<int>(node_ids_.size());
    write_int(n);
    for (int i = 0; i < n; ++i) {
        write_int(node_ids_[i]);
        for (int d = 0; d < dim_; ++d)
            write_float(node_vectors_[i][d]);
        write_int(node_levels_[i]);
    }

    // Deleted set
    int n_del = static_cast<int>(deleted_.size());
    write_int(n_del);
    for (int id : deleted_)
        write_int(id);

    // Layers
    int n_layers = static_cast<int>(layers_.size());
    write_int(n_layers);
    for (int lv = 0; lv < n_layers; ++lv) {
        int n_nodes = static_cast<int>(layers_[lv].size());
        write_int(n_nodes);
        for (int ni = 0; ni < n_nodes; ++ni) {
            int n_nb = static_cast<int>(layers_[lv][ni].size());
            write_int(n_nb);
            for (int nb : layers_[lv][ni])
                write_int(nb);
        }
    }
}

void HNSW::load(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file for reading: " + filepath);

    auto read_int = [&]() {
        int v; f.read(reinterpret_cast<char*>(&v), sizeof(v)); return v;
    };
    auto read_float = [&]() {
        float v; f.read(reinterpret_cast<char*>(&v), sizeof(v)); return v;
    };
    auto read_str = [&]() {
        int len = read_int();
        std::string s(len, '\0');
        f.read(&s[0], len);
        return s;
    };

    int version = read_int();
    if (version != FILE_VERSION)
        throw std::runtime_error("Unsupported file version: " + std::to_string(version));

    dim_ = read_int();
    metric_ = read_str();
    M_ = read_int();
    ef_construction_ = read_int();
    ef_search_ = read_int();
    entry_point_ = read_int();
    cur_max_level_ = read_int();

    // Nodes
    int n = read_int();
    node_ids_.resize(n);
    node_vectors_.resize(n, std::vector<float>(dim_));
    node_levels_.resize(n);
    node_norms_.assign(n, 0.0f);
    id_to_idx_.clear();
    for (int i = 0; i < n; ++i) {
        node_ids_[i] = read_int();
        for (int d = 0; d < dim_; ++d)
            node_vectors_[i][d] = read_float();
        node_levels_[i] = read_int();
        id_to_idx_[node_ids_[i]] = i;
        if (metric_ == "cosine")
            node_norms_[i] = dot_product(node_vectors_[i], node_vectors_[i]);
    }

    // Deleted set
    deleted_.clear();
    int n_del = read_int();
    for (int i = 0; i < n_del; ++i)
        deleted_.insert(read_int());

    // Layers
    int n_layers = read_int();
    layers_.resize(n_layers);
    for (int lv = 0; lv < n_layers; ++lv) {
        int n_nodes = read_int();
        layers_[lv].resize(n_nodes);
        for (int ni = 0; ni < n_nodes; ++ni) {
            int n_nb = read_int();
            layers_[lv][ni].resize(n_nb);
            for (int j = 0; j < n_nb; ++j)
                layers_[lv][ni][j] = read_int();
        }
    }
}
