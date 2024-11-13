#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <mkl.h>
#include <iomanip>

#define eps 1e-10

using namespace sycl;

class CSRMatrix {
public:
    std::vector<int> row_ptr;   // Указатели на начало строк
    std::vector<int> col_ind;   // Индексы столбцов ненулевых элементов
    std::vector<double> values;  // Ненулевые значения
    int rows, cols, non_zero_el; // Размеры матрицы, количсетво не нулевых элементов

    CSRMatrix() : CSRMatrix(0, 0) {}

    CSRMatrix(int rows, int cols) : rows(rows), cols(cols), non_zero_el(0) {
        row_ptr.resize(rows + 1, 0);
    }

    CSRMatrix(int r, int c, const std::vector<int>& rp, const std::vector<int>& ci, const std::vector<double>& v, int nze)
        : rows(r), cols(c), row_ptr(rp), col_ind(ci), values(v), non_zero_el(nze) {
            row_ptr.resize(rows + 1, 0);
    }

    // Функция для чтения матрицы из файла
    static CSRMatrix readFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Не удалось открыть файл: " + filename);
        }

        int rows, cols;
        file >> rows >> cols;

        CSRMatrix matrix(rows, cols);

        for (int i = 0; i <= rows; ++i) {
            file >> matrix.row_ptr[i];
        }

        int non_zero_elements = matrix.row_ptr.back();  // Количество ненулевых элементов
        matrix.col_ind.resize(non_zero_elements);
        matrix.values.resize(non_zero_elements);

        for (int i = 0; i < non_zero_elements; ++i) {
            file >> matrix.col_ind[i];
        }

        for (int i = 0; i < non_zero_elements; ++i) {
            file >> matrix.values[i];
        }

        file.close();
        return matrix;
    }

    // Функция для вывода матрицы
    void print() const {
        std::cout << "Row pointers: ";
        for (auto r : row_ptr) {
            std::cout << r << " ";
        }
        std::cout << "\nColumn indices: ";
        for (auto c : col_ind) {
            std::cout << c << " ";
        }
        std::cout << "\nValues: ";
        for (auto v : values) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

     CSRMatrix transpose() const {
        // Новые размеры
        int transposed_rows = cols;
        int transposed_cols = rows;
        int non_z_e = non_zero_el;


        // Количество ненулевых элементов в каждом столбце исходной матрицы (т.е. строке транспонированной)
        std::vector<int> row_count(transposed_rows, 0);
        
        // Подсчет ненулевых элементов в каждом столбце исходной матрицы
        for (int col : col_ind) {
            row_count[col]++;
        }

        // Построение нового row_ptr для транспонированной матрицы
        std::vector<int> transposed_row_ptr(transposed_rows + 1, 0);
        for (int i = 0; i < transposed_rows; ++i) {
            transposed_row_ptr[i + 1] = transposed_row_ptr[i] + row_count[i];
        }

        // Массивы для хранения индексов и значений транспонированной матрицы
        std::vector<int> transposed_col_ind(col_ind.size());
        std::vector<double> transposed_values(values.size());

        // Временный массив для отслеживания позиций вставки для каждого столбца (строки в транспонированной)
        std::vector<int> current_pos = transposed_row_ptr;

        // Заполнение новых col_ind и values
        for (int i = 0; i < rows; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                int col = col_ind[j];
                int dest_pos = current_pos[col];

                transposed_col_ind[dest_pos] = i;      // Строка в исходной матрице становится столбцом
                transposed_values[dest_pos] = values[j]; // Переносим значение

                current_pos[col]++;  // Обновляем позицию для следующего ненулевого элемента
            }
        }

        return CSRMatrix(transposed_rows, transposed_cols, transposed_row_ptr, transposed_col_ind, transposed_values, non_z_e);
    }
};

auto nvidia_selector = [](const device& dev) {
    const std::string name = dev.get_info<info::device::name>();
    if (name.find("NVIDIA") != std::string::npos) {
        return 1;  // Максимальный приоритет для NVIDIA устройств
    }
    return -1;  // Отбрасываем другие устройства
};

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

CSRMatrix csr_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix &C, queue& q);

CSRMatrix read_matrix_from_file(const std::string& filename);

void Available_platforms();

void sparse_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix &C, queue& q);