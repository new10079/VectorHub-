#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hnsw/hnsw.h"

namespace py = pybind11;

PYBIND11_MODULE(_vectorhub, m) {
    m.doc() = "VectorHub: lightweight vector database with HNSW indexing";

    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, const std::string&, int, int, int>(),
             py::arg("dim"),
             py::arg("metric") = "l2",
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef_search") = 50,
             "Initialize HNSW index\n\n"
             "Args:\n"
             "    dim: Vector dimension\n"
             "    metric: Distance metric ('l2' or 'cosine')\n"
             "    M: Max connections per node per layer (default 16)\n"
             "    ef_construction: Candidate pool size during build (default 200)\n"
             "    ef_search: Candidate pool size during search (default 50)")

        .def("insert", &HNSW::insert,
             py::arg("id"), py::arg("vector"),
             "Insert a single vector with id")

        .def("insert_batch", &HNSW::insert_batch,
             py::arg("ids"), py::arg("vectors"),
             "Insert multiple vectors with ids")

        .def("search", &HNSW::search,
             py::arg("query"), py::arg("k"),
             "HNSW approximate k-nearest-neighbor search.\n"
             "Returns list of (id, distance) tuples sorted by distance.")

        .def("brute_force_search", &HNSW::brute_force_search,
             py::arg("query"), py::arg("k"),
             "Exact brute-force k-nearest-neighbor search (for benchmarking / recall evaluation).\n"
             "Returns list of (id, distance) tuples sorted by distance.")

        .def("remove", &HNSW::remove,
             py::arg("id"),
             "Lazily delete a vector by id. Deleted ids are excluded from future searches.")

        .def("save", &HNSW::save,
             py::arg("filepath"),
             "Save the index to a binary file.")

        .def("load", &HNSW::load,
             py::arg("filepath"),
             "Load the index from a binary file (replaces current contents).")

        .def("set_ef_search", &HNSW::set_ef_search,
             py::arg("ef"),
             "Adjust ef_search at query time to trade recall for speed.")

        .def_property_readonly("dim",    &HNSW::dim)
        .def_property_readonly("metric", &HNSW::metric)
        .def_property_readonly("M",      &HNSW::get_M)
        .def_property_readonly("ef_construction", &HNSW::get_ef_construction)
        .def_property_readonly("ef_search",       &HNSW::get_ef_search)
        .def_property_readonly("size",   &HNSW::size,
             "Number of live (non-deleted) vectors in the index.");
}
