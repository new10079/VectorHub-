#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <limits>

class HNSW {
public:
    HNSW(int dim,
         const std::string& metric = "l2",
         int M = 16,
         int ef_construction = 200,
         int ef_search = 50);

    void insert(int id, const std::vector<float>& vector);
    void insert_batch(const std::vector<int>& ids,
                      const std::vector<std::vector<float>>& vectors);

    // HNSW graph search
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) const;

    // Brute-force exact search (baseline / recall benchmark)
    std::vector<std::pair<int, float>> brute_force_search(const std::vector<float>& query, int k) const;

    // Lazy delete - marks id as deleted; subsequent searches skip it
    void remove(int id);

    // Persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

    // Accessors
    int dim() const { return dim_; }
    std::string metric() const { return metric_; }
    int get_M() const { return M_; }
    int get_ef_construction() const { return ef_construction_; }
    int get_ef_search() const { return ef_search_; }
    void set_ef_search(int ef) { ef_search_ = ef; }
    int size() const;

private:
    int dim_;
    std::string metric_;
    int M_;
    int ef_construction_;
    int ef_search_;

    // Node storage: internal index -> (id, vector, level)
    std::vector<int> node_ids_;
    std::vector<std::vector<float>> node_vectors_;
    std::vector<int> node_levels_;   // level each node was assigned at (re)insertion time
    // Cached squared L2 norm of each node's vector, kept in sync with
    // node_vectors_ (only populated/used when metric_ == "cosine"). Lets
    // node-vs-node cosine distance (select_neighbors, pruning) skip
    // recomputing a node's own norm every time it's compared against.
    std::vector<float> node_norms_;
    std::unordered_map<int, int> id_to_idx_;
    std::unordered_set<int> deleted_;

    // Graph: layers_[level][internal_idx] = neighbor internal_idx list
    std::vector<std::vector<std::vector<int>>> layers_;

    // HNSW entry point (internal idx), -1 if empty
    int entry_point_;
    int cur_max_level_;

    mutable std::mt19937 rng_;

    float distance(const std::vector<float>& a, const std::vector<float>& b) const;
    float l2_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    float cosine_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    float dot_product(const std::vector<float>& a, const std::vector<float>& b) const;

    // Distance between two already-inserted nodes. Equivalent to
    // distance(node_vectors_[i], node_vectors_[j]) but for cosine reuses the
    // cached node_norms_ instead of recomputing each node's own norm --
    // matters in select_neighbors()/pruning where the same selected/kept
    // node is compared against many candidates.
    float node_distance(int i, int j) const;

    int random_level() const;

    // Search ef candidates at a single layer; returns max-heap of (dist, idx)
    // (max-heap so we can efficiently maintain ef closest)
    std::priority_queue<std::pair<float,int>>
    search_layer(const std::vector<float>& query,
                 int ep_idx,
                 int ef,
                 int level) const;

    std::vector<int> select_neighbors(const std::vector<float>& vec,
                                      const std::priority_queue<std::pair<float,int>>& candidates,
                                      int M_max) const;

    // Links node `idx` (vector already stored in node_vectors_[idx]) into the
    // graph at levels [0, node_level], exactly as a freshly-inserted node
    // would be. Grows layers_ as needed and updates entry_point_/cur_max_level_.
    // Shared by fresh inserts and by re-inserts of a previously-deleted id.
    void connect_new_node(int idx, const std::vector<float>& vector, int node_level);

    // Removes all of node `idx`'s existing edges (its own neighbor lists and
    // the reverse references held by its neighbors) at every level up to
    // old_level. Used before re-linking a reinserted node so stale edges
    // from its previous position don't linger in the graph.
    void disconnect_node(int idx, int old_level);

    static constexpr int FILE_VERSION = 3;
};
