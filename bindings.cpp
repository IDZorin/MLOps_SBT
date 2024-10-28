// bindings.cpp
#include "LinearAlgebra.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(linalg_core, m) {
    m.doc() = R"doc(
        Python bindings for LinearAlgebra library
    )doc";

    py::class_<LinearAlgebra>(m, "LinearAlgebra")
        .def_static("transpose", &LinearAlgebra::transpose, R"doc(
            Transpose a matrix.

            Parameters:
                matrix : list of list of float
                    The matrix to transpose.

            Returns:
                list of list of float
                    The transposed matrix.
        )doc");
}