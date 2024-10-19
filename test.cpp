#include "func.hpp"

int main() 
{
    std::cout << "Enter file name: ";
    std::string filename;
    std::cin >> filename;
    CSRMatrix A = read_matrix_from_file(filename);
    CSRMatrix B = read_matrix_from_file(filename);
    CSRMatrix AT;
    sparse_matrix_t A_csr_mkl = nullptr;
    sparse_matrix_t B_csr_mkl = nullptr;
    sparse_matrix_t C_csr_mkl = nullptr;
    sparse_matrix_t AT_mkl = nullptr;

    sparse_status_t status = mkl_sparse_d_create_csr(&A_csr_mkl, SPARSE_INDEX_BASE_ZERO, 
                            A.rows, A.cols, 
                            A.row_ptr.data(), A.row_ptr.data() + 1, 
                            A.col_ind.data(), A.values.data());

    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Ошибка при создании CSR матрицы\n");
        return -1;
    }

    status = mkl_sparse_d_create_csr(&B_csr_mkl, SPARSE_INDEX_BASE_ZERO, 
                            B.rows, B.cols, 
                            B.row_ptr.data(), B.row_ptr.data() + 1, 
                            B.col_ind.data(), B.values.data());
    
    if (status != SPARSE_STATUS_SUCCESS) {
        printf("Ошибка при создании CSR матрицы\n");
        return -1;
    }

    MKL_INT *rows_start, *rows_end, *columns;
    double *values;
    sparse_index_base_t indexing;
    MKL_INT rows, cols;

    // Создаем очередь для выполнения на устройстве (NVIDIA GPU)
    try {
        // Создаем очередь для CPU
        queue cpu_queue(cpu_selector_v, [](exception_list e_list) {
            for (std::exception_ptr const& e : e_list) {
                try {
                    std::rethrow_exception(e);
                } catch (std::exception const& e) {
                    std::cout << "CPU exception: " << e.what() << std::endl;
                }
            }
        });

        // Создаем очередь для NVIDIA GPU
        queue gpu_queue(nvidia_selector, [](exception_list e_list) {
            for (std::exception_ptr const& e : e_list) {
                try {
                    std::rethrow_exception(e);
                } catch (std::exception const& e) {
                    std::cout << "GPU exception: " << e.what() << std::endl;
                }
            }
        });

        // sparse_status_t status = mkl_sparse_d_create_csr(&A_csr_mkl, SPARSE_INDEX_BASE_ZERO, 
        //                         A.rows, A.cols, 
        //                         A.row_ptr.data(), A.row_ptr.data() + 1, 
        //                         A.col_ind.data(), A.values.data());

        // if (status != SPARSE_STATUS_SUCCESS) {
        //     printf("Ошибка при создании CSR матрицы\n");
        //     return -1;
        // }

        // status = mkl_sparse_d_create_csr(&B_csr_mkl, SPARSE_INDEX_BASE_ZERO, 
        //                         B.rows, B.cols, 
        //                         B.row_ptr.data(), B.row_ptr.data() + 1, 
        //                         B.col_ind.data(), B.values.data());
        
        // if (status != SPARSE_STATUS_SUCCESS) {
        //     printf("Ошибка при создании CSR матрицы\n");
        //     return -1;
        // }

        auto start = std::chrono::high_resolution_clock::now();
        CSRMatrix C_csr_cpu;
        sparse_matrix_multiply(A, B, C_csr_cpu, cpu_queue);
        auto end = std::chrono::high_resolution_clock::now();
        auto res = std::chrono::duration<double>(end - start).count();
        std::cout << "CSR CPU result: " << res << std::endl;

        start = std::chrono::high_resolution_clock::now();
        CSRMatrix C_csr_gpu;
        sparse_matrix_multiply(A, B, C_csr_gpu, gpu_queue);
        end = std::chrono::high_resolution_clock::now();
        res = std::chrono::duration<double>(end - start).count();
        std::cout << "CSR GPU result: " << res << std::endl;

        start = std::chrono::high_resolution_clock::now();
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A_csr_mkl, B_csr_mkl, &C_csr_mkl);
        end = std::chrono::high_resolution_clock::now();
        auto mkl_res = std::chrono::duration<double>(end - start).count();
        std::cout << "MKL result: " << mkl_res << " seconds" << std::endl;

        /*// Проверяем корректность вычислений
        if (C_csr_gpu.row_ptr != C_csr_cpu.row_ptr && C_csr_gpu.col_ind != C_csr_cpu.col_ind && C_csr_gpu.values != C_csr_cpu.values
            && C_csr_mkl -> rowIndex != C_csr_gpu.row_ptr && C_csr_mkl -> columns != C_csr_gpu.col_ind && C_csr_mkl -> values != C_csr_gpu.values){
            std::cout << "Results aren't correct!" << std::endl;
        }
        else
        {
            std::cout << "Results are correct!" << std::endl;
        }*/
        
        mkl_sparse_d_export_csr(C_csr_mkl, &indexing, &rows, &cols, &rows_start, &rows_end, &columns, &values);

        // Сравнение результатов между MKL и SYCL
        bool correct = true;
        for (int i = 0; i < C_csr_cpu.row_ptr.size(); i++) {
            if (rows_start[i] != C_csr_cpu.row_ptr[i]) {
                correct = false;
                std::cout << "R: cpu " << C_csr_cpu.row_ptr[i] << " mkl " << rows_start[i] << std::endl;
                break;
            }
        }
        for (int i = 0; i < C_csr_cpu.col_ind.size(); i++) {
            if (columns[i] != C_csr_cpu.col_ind[i]) {
                correct = false;
                std::cout << "C: cpu " << C_csr_cpu.col_ind[i] << " mkl " << columns[i] << std::endl;
                break;
            }
        }
        for (int i = 0; i < C_csr_cpu.values.size(); i++) {
            if (values[i] != C_csr_cpu.values[i]) {
                correct = false;
                std::cout << "V: cpu " << C_csr_cpu.values[i] << " mkl " << values[i] << std::endl;
                break;
            }
        }

        if (correct) {
            std::cout << "Results are correct!" << std::endl;
        } else {
            std::cout << "Results aren't correct!" << std::endl;
        }

        if (A_csr_mkl != nullptr) {
            mkl_sparse_destroy(A_csr_mkl);
        }
        if (B_csr_mkl != nullptr) {
            mkl_sparse_destroy(B_csr_mkl);
        }
        if (C_csr_mkl != nullptr) {
            mkl_sparse_destroy(C_csr_mkl);
        }

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}