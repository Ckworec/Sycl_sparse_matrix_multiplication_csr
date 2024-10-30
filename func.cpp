#include "func.hpp"
void sparse_matrix_multiply(const CSRMatrix& A, const CSRMatrix& B1, CSRMatrix& C, queue& q) {
    if (A.cols != B1.rows) {
        throw std::runtime_error("Размеры матриц не совпадают для умножения.");
    }

    // Результирующая матрица
    C.rows = A.rows;
    C.cols = B1.cols;
    int ro = A.rows;

    CSRMatrix B = B1.transpose();

    std::vector<int> row_ptr_C(A.rows + 1, 0);
    std::vector<int> col_ind_C;
    std::vector<double> values_C;
    int ro2 = B.rows;

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

        h.parallel_for(range<1>(ro - 1), [=](id<1> ind) {
            int i = ind[0] + 1;

            acc_row_ptr_C[i] = acc_row_ptr_C[i - 1];
            
            for (int bpos = 1; bpos < ro2; ++bpos) {
                int sum = 0;
                for (int r = acc_row_ptr_A[i - 1]; r < acc_row_ptr_A[i]; ++r) {
                    for (int j = acc_row_ptr_B[bpos - 1]; j < acc_row_ptr_B[bpos]; ++j) {
                        if (acc_col_ind_B[j] == acc_col_ind_A[r]) {
                            sum += acc_values_A[r] * acc_values_B[j];
                        }
                    }
                }
                if (sum!= 0) {
                    acc_row_ptr_C[i]++;
                    acc_col_ind_C[acc_non_zero_el_C[0]] = bpos;
                    acc_values_C[acc_non_zero_el_C[0]] = sum;
                    acc_non_zero_el_C[0]++;
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