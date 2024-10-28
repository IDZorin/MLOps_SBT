// test_linalg.cpp
#include <iostream>
#include "LinearAlgebra.h"

int main() {
    // Пример матрицы для тестирования транспонирования
    std::vector<std::vector<float>> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    // Ожидаемый результат транспонирования
    std::vector<std::vector<float>> expected_transpose = {
        {1.0, 4.0},
        {2.0, 5.0},
        {3.0, 6.0}
    };

    // Выполняем транспонирование с помощью функции
    std::vector<std::vector<float>> result = LinearAlgebra::transpose(matrix);

    // Проверяем результат
    if (result == expected_transpose) {
        std::cout << "Transpose test passed!" << std::endl;
    } else {
        std::cout << "Transpose test failed." << std::endl;
    }

    return 0;
}
