#include "func.hpp"

// Функция умножения двух разреженных матриц в формате CSR
/*CSRMatrix sparse_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B, queue& q) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Размеры матриц не совпадают для умножения.");
    }

    // Результирующая матрица
    CSRMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;

    std::vector<int> row_ptr_C(A.rows + 1, 0);
    std::vector<int> col_ind_C;
    std::vector<double> values_C;

    // Параллельное умножение разреженных матриц
    q.submit([&](handler& h) {
        // Используем буферы для передачи данных на устройство
        buffer<int> buf_row_ptr_A(A.row_ptr.data(), range<1>(A.row_ptr.size()));
        buffer<int> buf_col_ind_A(A.col_ind.data(), range<1>(A.col_ind.size()));
        buffer<double> buf_values_A(A.values.data(), range<1>(A.values.size()));

        buffer<int> buf_row_ptr_B(B.row_ptr.data(), range<1>(B.row_ptr.size()));
        buffer<int> buf_col_ind_B(B.col_ind.data(), range<1>(B.col_ind.size()));
        buffer<double> buf_values_B(B.values.data(), range<1>(B.values.size()));

        buffer<int> buf_row_ptr_C(row_ptr_C.data(), range<1>(row_ptr_C.size()));

        // Использование аксессоров
        auto acc_row_ptr_A = buf_row_ptr_A.get_access<access::mode::read>(h);
        auto acc_col_ind_A = buf_col_ind_A.get_access<access::mode::read>(h);
        auto acc_values_A = buf_values_A.get_access<access::mode::read>(h);

        auto acc_row_ptr_B = buf_row_ptr_B.get_access<access::mode::read>(h);
        auto acc_col_ind_B = buf_col_ind_B.get_access<access::mode::read>(h);
        auto acc_values_B = buf_values_B.get_access<access::mode::read>(h);

        auto acc_row_ptr_C = buf_row_ptr_C.get_access<access::mode::write>(h);

        // Параллельный цикл по строкам матрицы A
        h.parallel_for(range<1>(A.rows), [=](id<1> i) {
            int row_start_A = acc_row_ptr_A[i];
            int row_end_A = acc_row_ptr_A[i + 1];

            for (int j = row_start_A; j < row_end_A; ++j) {
                int col_A = acc_col_ind_A[j];
                double val_A = acc_values_A[j];

                int row_start_B = acc_row_ptr_B[col_A];
                int row_end_B = acc_row_ptr_B[col_A + 1];

                for (int k = row_start_B; k < row_end_B; ++k) {
                    int col_B = acc_col_ind_B[k];
                    double val_B = acc_values_B[k];

                    acc_row_ptr_C[i]++; // Увеличиваем количество ненулевых элементов
                }
            }
            });
        }).wait();

    C.row_ptr = row_ptr_C;
    C.col_ind = col_ind_C;
    C.values = values_C;

    return C;
}*/

void sparse_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B1, CSRMatrix& C, queue& q) {
    if (A.cols != B1.rows) {
        throw std::runtime_error("Размеры матриц не совпадают для умножения.");
    }

    // Результирующая матрица
    C.rows = A.rows;
    C.cols = B1.cols;

    CSRMatrix B = B1.transpose();

    std::vector<int> row_ptr_C(A.rows + 1, 0);
    std::vector<int> col_ind_C;
    std::vector<double> values_C;

    // Используем буферы для передачи данных на устройство
    buffer<int> buf_row_ptr_A(A.row_ptr.data(), range<1>(A.row_ptr.size()));
    buffer<int> buf_col_ind_A(A.col_ind.data(), range<1>(A.col_ind.size()));
    buffer<double> buf_values_A(A.values.data(), range<1>(A.values.size()));
    buffer<int, 1> buf_non_zero_el_A(&A.non_zero_el, sycl::range<1>(1));

    buffer<int> buf_row_ptr_B(B.row_ptr.data(), range<1>(B.row_ptr.size()));
    buffer<int> buf_col_ind_B(B.col_ind.data(), range<1>(B.col_ind.size()));
    buffer<double> buf_values_B(B.values.data(), range<1>(B.values.size()));
    buffer<int, 1> buf_non_zero_el_B(&B.non_zero_el, sycl::range<1>(1));

    buffer<int> buf_row_ptr_C(row_ptr_C.data(), range<1>(row_ptr_C.size()));

    // Создаем временные буферы для хранения индексов и значений ненулевых элементов результирующей матрицы
    buffer<int> buf_col_ind_C(0, range<1>(0));  // Будем динамически изменять размер
    buffer<double> buf_values_C(0, range<1>(0));  // Будем динамически изменять размер
    buffer<int, 1> buf_non_zero_el_C(&C.non_zero_el, sycl::range<1>(1));

    // Параллельное умножение разреженных матриц
    q.submit([&](handler& h) {
        int nze = 0;

        // Использование аксессоров
        auto acc_row_ptr_A = buf_row_ptr_A.get_access<access::mode::read>(h);
        auto acc_col_ind_A = buf_col_ind_A.get_access<access::mode::read>(h);
        auto acc_values_A = buf_values_A.get_access<access::mode::read>(h);
        auto acc_non_zero_el_A = buf_non_zero_el_A.get_access<sycl::access::mode::read>(h);

        auto acc_row_ptr_B = buf_row_ptr_B.get_access<access::mode::read>(h);
        auto acc_col_ind_B = buf_col_ind_B.get_access<access::mode::read>(h);
        auto acc_values_B = buf_values_B.get_access<access::mode::read>(h);
        auto acc_non_zero_el_B = buf_non_zero_el_B.get_access<sycl::access::mode::read>(h);

        auto acc_row_ptr_C = buf_row_ptr_C.get_access<access::mode::write>(h);
        auto acc_col_ind_C = buf_col_ind_C.get_access<access::mode::write>(h);
        auto acc_values_C = buf_values_C.get_access<access::mode::write>(h);
        auto acc_non_zero_el_C = buf_non_zero_el_C.get_access<sycl::access::mode::read_write>(h);

        acc_row_ptr_C[0] = 0;

        h.parallel_for(range<1>(A.rows - 1), [=](id<1> ind) {
            /*int row_start_A = acc_row_ptr_A[i];
            int row_end_A = acc_row_ptr_A[i + 1];

            // Переменная для подсчета ненулевых элементов в текущей строке C
            int non_zero_count = 0;

            for (int j = row_start_A; j < row_end_A; ++j) {
                int col_A = acc_col_ind_A[j];
                double val_A = acc_values_A[j];

                int row_start_B = acc_row_ptr_B[col_A];
                int row_end_B = acc_row_ptr_B[col_A + 1];

                for (int k = row_start_B; k < row_end_B; ++k) {
                    int col_B = acc_col_ind_B[k];
                    double val_B = acc_values_B[k];

                    // Суммируем произведение
                    double product = val_A * val_B;

                    // Добавляем ненулевой элемент в C
                    bool found = false;
                    for (int idx = acc_row_ptr_C[i]; idx < acc_row_ptr_C[i + 1]; ++idx) {
                        if (acc_col_ind_C[idx] == col_B) {
                            acc_values_C[idx] += product; // Накопление результата
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        // Если элемент не найден, добавляем новый ненулевой элемент
                        int new_index = non_zero_count + acc_row_ptr_C[i + 1];
                        acc_col_ind_C[new_index] = col_B;
                        acc_values_C[new_index] = product;
                        non_zero_count++;
                    }
                }
            }

            // Обновляем указатели в row_ptr_C
            acc_row_ptr_C[i + 1] += non_zero_count;

            int r = acc_row_ptr_A[i];*/

            int i = ind[0] + 1;

            acc_row_ptr_C[i] = acc_row_ptr_C[i - 1];

            for (int bpos = 1; bpos < B.rows; ++bpos) {
                int sum = 0;
                for (int r = acc_row_ptr_A[i - 1]; r < acc_row_ptr_A[i]; ++r) {
                    auto it = std::find(acc_col_ind_B[acc_row_ptr_B[bpos - 1]], acc_col_ind_A[acc_row_ptr_B[i]], acc_col_ind_A[r]);
                    if (it != acc_col_ind_A[acc_row_ptr_B[i]]){
                        sum += acc_values_A[r] * acc_values_B[it - acc_col_ind_B[0]];
                    }
                }
                if (sum!= 0) {
                    acc_row_ptr_C[i]++;
                    acc_col_ind_C[acc_non_zero_el_C[0]] = bpos;
                    acc_values_C[acc_non_zero_el_C[0]] = sum;
                    acc_non_zero_el_C[0]++;
                }
            }

            /*for (int bpos = 0; bpos < B.rows; ++bpos) {
                int c = acc_row_ptr_B[bpos];

                int tempa = i;
                int tempb = bpos;

                int sum = 0;

                while (tempa < acc_non_zero_el_A && acc_row_ptr_A[tempa] == r &&
                        tempb < B.non_zero_el && acc_row_ptr_B[tempb] == c) {
                    if (acc_col_ind_A[tempa] < acc_col_ind_B[tempb])
                        tempa++;
                    else if (acc_col_ind_A[tempa] > acc_col_ind_B[tempb])
                        tempb++;
                    else
                        sum += acc_values_A[tempa++] * acc_values_B[tempb++];
                }

                if (sum != 0){
                    acc_row_ptr_C[A.rows] = r;
                    acc_col_ind_C[acc_non_zero_el_C[0]] = c;
                    acc_values_C[acc_non_zero_el_C[0]] = sum;
                    acc_non_zero_el_C[0]++;
                }
            }

            int apos = acc_row_ptr_A[i];
            int r = i[0];  // row of A and result C

            while (apos < acc_row_ptr_A[i + 1]) {
                int a_col = acc_col_ind_A[apos];
                float a_val = acc_values_A[apos];

                for (int bpos = acc_row_ptr_B[a_col]; bpos < acc_row_ptr_B[a_col + 1]; ++bpos) {
                    int b_col = acc_col_ind_B[bpos];
                    float b_val = acc_values_B[bpos];

                    float sum = 0;
                    int tempa = apos;
                    int tempb = bpos;

                    // Multiply non-zero values
                    while (tempa < acc_row_ptr_A[i + 1] && tempb < acc_row_ptr_B[a_col + 1]) {
                        if (acc_col_ind_A[tempa] < acc_col_ind_B[tempb]) {
                            tempa++;
                        } else if (acc_col_ind_A[tempa] > acc_col_ind_B[tempb]) {
                            tempb++;
                        } else {
                            sum += acc_values_A[tempa++] * acc_values_B[tempb++];
                        }
                    }

                    if (sum != 0) {
                        // Store results in matrix C
                        acc_row_ptr_C[i] = r;
                        acc_col_ind_C[i] = b_col;
                        acc_values_C[i] = sum;
                    }
                }
                ++apos;
            }*/
        });
    }).wait();
   
}

// Функция для чтения матрицы в формате CSR из файла
CSRMatrix read_matrix_from_file(const std::string& filename) {
    CSRMatrix matrix;
    int non_zero_elements;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    // Считываем количество строк и столбцов
    file >> matrix.rows >> non_zero_elements;
    matrix.cols = matrix.rows;
    matrix.non_zero_el = non_zero_elements;

    // Считываем массив row_ptr
    matrix.row_ptr.resize(matrix.rows + 1);
    for (int i = 0; i <= matrix.rows; ++i) {
        file >> matrix.row_ptr[i];
    }

    // Считываем массив col_ind
    matrix.col_ind.resize(non_zero_elements);
    for (int i = 0; i < non_zero_elements; ++i) {
        file >> matrix.col_ind[i];
    }

    // Считываем массив values
    matrix.values.resize(non_zero_elements);
    for (int i = 0; i < non_zero_elements; ++i) {
        file >> matrix.values[i];
    }

    file.close();
    return matrix;
}

void Available_platforms()
{
    // Получаем платформы
    std::vector<platform> platforms = platform::get_platforms();

    for (const auto& plat : platforms) {
        std::cout << "Platform: " << plat.get_info<info::platform::name>() << "\n";

        // Получаем устройства на каждой платформе
        std::vector<device> devices = plat.get_devices();
        for (const auto& dev : devices) {
            std::cout << "  Device: " << dev.get_info<info::device::name>() << "\n";
            std::cout << "    Type: "
                << (dev.is_gpu() ? "GPU" : (dev.is_cpu() ? "CPU" : "Other"))
                << "\n";
            std::cout << "    Max Compute Units: "
                << dev.get_info<info::device::max_compute_units>()
                << "\n";
            std::cout << "    Global Memory Size: "
                << dev.get_info<info::device::global_mem_size>()
                << " bytes\n";
        }
    }

    std::cout << "\n" << std::endl;
}

/*void multiply(sparse_matrix b)
{
    if (col != b.row)
    {
        cout << "Can't multiply, Invalid dimensions";
        return;
    }

    // транспонировать b, чтобы сравнить значения строки и столбца и добавить их в конце
    b = b.transpose();
    int apos, bpos;

    // результирующая матрица размерности row * b.col однако b был транспонирован, следовательно row X b.row
    sparse_matrix result(row, b.row);

    // iterate over all elements of A
    for (apos = 0; apos < len;)
    {
        // текущая строка матрицы результатов
        int r = data[apos][0];

        // iterate over all elements of B
        for (bpos = 0; bpos < b.len;)
        {
            // текущий столбец матрицы результата data[,0], используемый как b, транспонирован
            int c = b.data[bpos][0];

            // временные указатели, созданные для сложения всех умноженных значений для получения текущего элемента матрицы результата
            int tempa = apos;
            int tempb = bpos;

            int sum = 0;

            // перебрать все элементы с одинаковым значением строки и столбца для вычисления результата[r]
            while (tempa < len && data[tempa][0] == r &&
                    tempb < b.len && b.data[tempb][0] == c)
            {
                if (data[tempa][1] < b.data[tempb][1])
                    // skip a
                    tempa++;

                else if (data[tempa][1] > b.data[tempb][1])
                    // skip b
                    tempb++;
                else
                    // тот же столбец, поэтому умножаем и увеличиваем
                    sum += data[tempa++][2] * b.data[tempb++][2];
            }

            // вставьте полученную сумму в result[r], если она не равна 0
            if (sum != 0)
                result.insert(r, c, sum);

            while (bpos < b.len && b.data[bpos][0] == c)
                // jump to next column
                bpos++;
        }
        while (apos < len && data[apos][0] == r)
            // jump to next row
            apos++;
    }
    result.print();
}*/
