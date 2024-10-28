// LinearAlgebra.cpp
#include "LinearAlgebra.h"

std::vector<std::vector<float>> LinearAlgebra::transpose(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) return {};

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[j][i] = matrix[i][j];

    return result;
}
