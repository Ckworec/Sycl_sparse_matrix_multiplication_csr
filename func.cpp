#include "func.hpp"

void sparse_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B1, CSRMatrix& C, queue& q) {
    if (A.cols != B1.rows) {
        throw std::runtime_error("Размеры матриц не совпадают для умножения.");
    }

    // Результирующая матрица
    C.rows = A.rows;
    C.cols = B1.cols;
    C.non_zero_el = 0;
    C.row_ptr.resize(A.rows + 1);
    int ro = A.rows;

    CSRMatrix B = B1.transpose();

    int ro2 = B.rows;

    // Используем буферы для передачи данных на устройство
    buffer<int, 1> buf_row_ptr_A(A.row_ptr.data(), range<1>(A.row_ptr.size()));
    buffer<int, 1> buf_col_ind_A(A.col_ind.data(), range<1>(A.col_ind.size()));
    buffer<double, 1> buf_values_A(A.values.data(), range<1>(A.values.size()));
    buffer<int, 1> buf_non_zero_el_A(&A.non_zero_el, range<1>(1));

    buffer<int, 1> buf_row_ptr_B(B.row_ptr.data(), range<1>(B.row_ptr.size()));
    buffer<int, 1> buf_col_ind_B(B.col_ind.data(), range<1>(B.col_ind.size()));
    buffer<double, 1> buf_values_B(B.values.data(), range<1>(B.values.size()));
    buffer<int, 1> buf_non_zero_el_B(&B.non_zero_el, range<1>(1));

    buffer<int, 1> buf_row_ptr_C(C.row_ptr.data(), range<1>(C.row_ptr.size()));

    // Параллельное умножение разреженных матриц
    q.submit([&](handler& h) {
        // Использование аксессоров
        accessor acc_row_ptr_A = buf_row_ptr_A.get_access<access::mode::read>(h);
        accessor acc_col_ind_A = buf_col_ind_A.get_access<access::mode::read>(h);
        accessor acc_values_A = buf_values_A.get_access<access::mode::read>(h);
        accessor acc_non_zero_el_A = buf_non_zero_el_A.get_access<access::mode::read>(h);

        accessor acc_row_ptr_B = buf_row_ptr_B.get_access<access::mode::read>(h);
        accessor acc_col_ind_B = buf_col_ind_B.get_access<access::mode::read>(h);
        accessor acc_values_B = buf_values_B.get_access<access::mode::read>(h);
        accessor acc_non_zero_el_B = buf_non_zero_el_B.get_access<access::mode::read>(h);

        accessor acc_row_ptr_C = buf_row_ptr_C.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(ro), [=](id<1> ind) {
            size_t i = ind[0] + 1;
            int k = 0;
            
            for (int bpos = 1; bpos < ro2 + 1; ++bpos) {
                double sum = 0;
                for (int r = acc_row_ptr_A[i - 1]; r < acc_row_ptr_A[i]; ++r) {
                    for (int j = acc_row_ptr_B[bpos - 1]; j < acc_row_ptr_B[bpos]; ++j) {
                        if (acc_col_ind_B[j] == acc_col_ind_A[r]) {
                            sum += acc_values_A[r] * acc_values_B[j];
                        }
                    }
                }
                // sycl::ext::oneapi::experimental::printf("Индекс: sum = %lf\n", sum);
                if (abs(sum) > eps) {
                    k ++;
                }
                // sycl::ext::oneapi::experimental::printf("Индекс: i = %d    bpos = %d    sum = %lf    k = %d\n", i, bpos, sum, k);
            }
            acc_row_ptr_C[i] = k;
            // sycl::ext::oneapi::experimental::printf("Индекс: acc_row_ptr_C[i] = %d\n", acc_row_ptr_C[i]);
        });
    }).wait();

    for (int k = 1; k < C.row_ptr.size(); k++) {
        C.row_ptr[k] += C.row_ptr[k - 1];
        // std::cout << C.row_ptr[k] << std::endl;
    }

    C.non_zero_el = C.row_ptr[C.row_ptr.size() - 1];

    // std::cout << C.non_zero_el << std::endl;

    C.col_ind.resize(C.non_zero_el);
    C.values.resize(C.non_zero_el);

    // Создаем временные буферы для хранения индексов и значений ненулевых элементов результирующей матрицы
    buffer<int, 1> buf_col_ind_C(C.col_ind.data(), range<1>(C.non_zero_el));  // Будем динамически изменять размер
    buffer<double, 1> buf_values_C(C.values.data(), range<1>(C.non_zero_el));  // Будем динамически изменять размер
    buffer<int, 1> buf_non_zero_el_C(&C.non_zero_el, range<1>(1));

    // Параллельное умножение разреженных матриц
    q.submit([&](handler& h) {
        // Использование аксессоров
        accessor acc_row_ptr_A = buf_row_ptr_A.get_access<access::mode::read>(h);
        accessor acc_col_ind_A = buf_col_ind_A.get_access<access::mode::read>(h);
        accessor acc_values_A = buf_values_A.get_access<access::mode::read>(h);
        accessor acc_non_zero_el_A = buf_non_zero_el_A.get_access<access::mode::read>(h);

        accessor acc_row_ptr_B = buf_row_ptr_B.get_access<access::mode::read>(h);
        accessor acc_col_ind_B = buf_col_ind_B.get_access<access::mode::read>(h);
        accessor acc_values_B = buf_values_B.get_access<access::mode::read>(h);
        accessor acc_non_zero_el_B = buf_non_zero_el_B.get_access<access::mode::read>(h);

        accessor acc_row_ptr_C = buf_row_ptr_C.get_access<access::mode::read_write>(h);
        accessor acc_col_ind_C = buf_col_ind_C.get_access<access::mode::write>(h);
        accessor acc_values_C = buf_values_C.get_access<access::mode::write>(h);
        accessor acc_non_zero_el_C = buf_non_zero_el_C.get_access<access::mode::read_write>(h);

        h.parallel_for(range<1>(ro), [=](id<1> ind) {
            size_t i = ind[0] + 1;
            int k = 0;

            // sycl::ext::oneapi::experimental::printf("Индекс: %d\n", ind[0] + 1);
            
            for (int bpos = 1; bpos < ro2 + 1; ++bpos) {
                double sum = 0;
                for (int r = acc_row_ptr_A[i - 1]; r < acc_row_ptr_A[i]; ++r) {
                    for (int j = acc_row_ptr_B[bpos - 1]; j < acc_row_ptr_B[bpos]; ++j) {
                        // sycl::ext::oneapi::experimental::printf("Индекс: i = %d    bpos = %d   r = %d   j = %d\n", ind[0] + 1, bpos, r, j);
                        if (acc_col_ind_B[j] == acc_col_ind_A[r]) {
                            sum += acc_values_A[r] * acc_values_B[j];
                            break;
                        }
                    }
                }
                if (abs(sum) > eps) {
                    // sycl::ext::oneapi::experimental::printf("Индекс: i - 1 = %d    bpos - 1 = %d   sum = %lf    k = %d\n", i - 1, bpos - 1, sum, k);
                    acc_col_ind_C[acc_row_ptr_C[i - 1] + k] = bpos - 1;
                    acc_values_C[acc_row_ptr_C[i - 1] + k] = sum;
                    k ++;
                }
            }
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