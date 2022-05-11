#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <thread>
#include <string.h>
#include <arm_neon.h>

#define PRINT_TIME(code) do { \
    auto start = system_clock::now(); \
    code \
    auto end   = system_clock::now(); \
    auto duration = duration_cast<microseconds>(end - start); \
    cout << "time spent: " << double(duration.count()) << "us" << endl; \
} while(0)

using namespace std;

using namespace chrono;

using vec = vector<int>; 

const int scale[] = {256, 512, 1024, 2048};
const string data_path("./data/");
const int num_threads = 8;
int block = 256;

int a[4194304] __attribute__((__aligned__((0x100))));
int b[4194304] __attribute__((__aligned__((0x100))));
int c[4194304] __attribute__((__aligned__((0x100))));
int b_transport[4194304] __attribute__((__aligned__((0x100))));

void transpose(const int size, int thread_id){
    int single_processing_line = size / num_threads;
    int start = single_processing_line * thread_id;
    int end = single_processing_line * (thread_id + 1);
    for(int i = start; i < end; ++i){
        for(int j = 0; j < size; ++j){
            b_transport[i*size + j] = b[j*size + i];
        }
    }
}

void multi_threads_gemm(const int size, int thread_id){                
    int single_processing_line = size / num_threads; 
    int start = single_processing_line * thread_id;
    int end = single_processing_line * (thread_id + 1);
    int i = start;
    int sum_00, sum_01, sum_02, sum_03;
    int sum_10, sum_11, sum_12, sum_13;
    int sum_20, sum_21, sum_22, sum_23;
    int sum_30, sum_31, sum_32, sum_33;
    int32x4_t sum_group_val_00, sum_group_val_10, sum_group_val_20, sum_group_val_30;
    int32x4_t sum_group_val_01, sum_group_val_11, sum_group_val_21, sum_group_val_31;
    int32x4_t sum_group_val_02, sum_group_val_12, sum_group_val_22, sum_group_val_32;
    int32x4_t sum_group_val_03, sum_group_val_13, sum_group_val_23, sum_group_val_33;
    int32x4_t a_vec_val_0, a_vec_val_1, a_vec_val_2, a_vec_val_3;
    int32x4_t b_vec_val_0, b_vec_val_1, b_vec_val_2, b_vec_val_3;
    int32x4_t * a_addr_init = (int32x4_t *)((int *)&a + start * size);
    int32x4_t * a_group_addr_now_0, * a_group_addr_now_1, * a_group_addr_now_2, * a_group_addr_now_3;
    int32x4_t * b_group_addr_now_0, * b_group_addr_now_1, * b_group_addr_now_2, * b_group_addr_now_3;
    int32x4_t * b_transport_addr;
    if(block > size)
        block = size;
    for(int m = 0; m < size; m += block) {
        int row_block_limit = m + block;
        for(int n = 0; n < size; n += block) {
            int32x4_t * a_addr = (int32x4_t *)((int *)a_addr_init + n);
            for(i = start; i < end; i += 4) {
                b_transport_addr = (int32x4_t *)((int *)&b_transport + m * size + n);
                for(int j = m; j < row_block_limit; j += 4) {
                    sum_group_val_00 = {0, 0, 0, 0}; sum_group_val_01 = {0, 0, 0, 0}; sum_group_val_02 = {0, 0, 0, 0}; sum_group_val_03 = {0, 0, 0, 0};
                    sum_group_val_10 = {0, 0, 0, 0}; sum_group_val_11 = {0, 0, 0, 0}; sum_group_val_12 = {0, 0, 0, 0}; sum_group_val_13 = {0, 0, 0, 0};
                    sum_group_val_20 = {0, 0, 0, 0}; sum_group_val_21 = {0, 0, 0, 0}; sum_group_val_22 = {0, 0, 0, 0}; sum_group_val_23 = {0, 0, 0, 0};
                    sum_group_val_30 = {0, 0, 0, 0}; sum_group_val_31 = {0, 0, 0, 0}; sum_group_val_32 = {0, 0, 0, 0}; sum_group_val_33 = {0, 0, 0, 0}; 
                    a_group_addr_now_0 = a_addr;
                    a_group_addr_now_1 = (int32x4_t *)((int *)a_addr + size);
                    a_group_addr_now_2 = (int32x4_t *)((int *)a_addr + 2 * size);
                    a_group_addr_now_3 = (int32x4_t *)((int *)a_addr + 3 * size);
                    b_group_addr_now_0 = b_transport_addr;
                    b_group_addr_now_1 = (int32x4_t *)((int *)b_transport_addr + size);
                    b_group_addr_now_2 = (int32x4_t *)((int *)b_transport_addr + 2 * size);
                    b_group_addr_now_3 = (int32x4_t *)((int *)b_transport_addr + 3 * size);
                    for(int k = 0; k < block; k += 4) {
                        a_vec_val_0 = *a_group_addr_now_0;
                        a_vec_val_1 = *a_group_addr_now_1;
                        a_vec_val_2 = *a_group_addr_now_2;
                        a_vec_val_3 = *a_group_addr_now_3;
                        ++a_group_addr_now_0;
                        ++a_group_addr_now_1;
                        ++a_group_addr_now_2;
                        ++a_group_addr_now_3;
                        b_vec_val_0 = *b_group_addr_now_0;
                        b_vec_val_1 = *b_group_addr_now_1;
                        b_vec_val_2 = *b_group_addr_now_2;
                        b_vec_val_3 = *b_group_addr_now_3;
                        ++b_group_addr_now_0;
                        ++b_group_addr_now_1;
                        ++b_group_addr_now_2;
                        ++b_group_addr_now_3;
                        sum_group_val_00 = vmlaq_s32(sum_group_val_00, b_vec_val_0, a_vec_val_0);
                        sum_group_val_01 = vmlaq_s32(sum_group_val_01, b_vec_val_1, a_vec_val_0);
                        sum_group_val_02 = vmlaq_s32(sum_group_val_02, b_vec_val_2, a_vec_val_0);
                        sum_group_val_03 = vmlaq_s32(sum_group_val_03, b_vec_val_3, a_vec_val_0);
                        sum_group_val_10 = vmlaq_s32(sum_group_val_10, b_vec_val_0, a_vec_val_1);
                        sum_group_val_11 = vmlaq_s32(sum_group_val_11, b_vec_val_1, a_vec_val_1);
                        sum_group_val_12 = vmlaq_s32(sum_group_val_12, b_vec_val_2, a_vec_val_1);
                        sum_group_val_13 = vmlaq_s32(sum_group_val_13, b_vec_val_3, a_vec_val_1);
                        sum_group_val_20 = vmlaq_s32(sum_group_val_20, b_vec_val_0, a_vec_val_2);
                        sum_group_val_21 = vmlaq_s32(sum_group_val_21, b_vec_val_1, a_vec_val_2);
                        sum_group_val_22 = vmlaq_s32(sum_group_val_22, b_vec_val_2, a_vec_val_2);
                        sum_group_val_23 = vmlaq_s32(sum_group_val_23, b_vec_val_3, a_vec_val_2);
                        sum_group_val_30 = vmlaq_s32(sum_group_val_30, b_vec_val_0, a_vec_val_3);
                        sum_group_val_31 = vmlaq_s32(sum_group_val_31, b_vec_val_1, a_vec_val_3);
                        sum_group_val_32 = vmlaq_s32(sum_group_val_32, b_vec_val_2, a_vec_val_3);
                        sum_group_val_33 = vmlaq_s32(sum_group_val_33, b_vec_val_3, a_vec_val_3);
                    }
                    sum_00 = sum_group_val_00[0] + sum_group_val_00[1] + sum_group_val_00[2] + sum_group_val_00[3];
                    sum_01 = sum_group_val_01[0] + sum_group_val_01[1] + sum_group_val_01[2] + sum_group_val_01[3];
                    sum_02 = sum_group_val_02[0] + sum_group_val_02[1] + sum_group_val_02[2] + sum_group_val_02[3];
                    sum_03 = sum_group_val_03[0] + sum_group_val_03[1] + sum_group_val_03[2] + sum_group_val_03[3];
                    sum_10 = sum_group_val_10[0] + sum_group_val_10[1] + sum_group_val_10[2] + sum_group_val_10[3];
                    sum_11 = sum_group_val_11[0] + sum_group_val_11[1] + sum_group_val_11[2] + sum_group_val_11[3];
                    sum_12 = sum_group_val_12[0] + sum_group_val_12[1] + sum_group_val_12[2] + sum_group_val_12[3];
                    sum_13 = sum_group_val_13[0] + sum_group_val_13[1] + sum_group_val_13[2] + sum_group_val_13[3];
                    sum_20 = sum_group_val_20[0] + sum_group_val_20[1] + sum_group_val_20[2] + sum_group_val_20[3];
                    sum_21 = sum_group_val_21[0] + sum_group_val_21[1] + sum_group_val_21[2] + sum_group_val_21[3];
                    sum_22 = sum_group_val_22[0] + sum_group_val_22[1] + sum_group_val_22[2] + sum_group_val_22[3];
                    sum_23 = sum_group_val_23[0] + sum_group_val_23[1] + sum_group_val_23[2] + sum_group_val_23[3];
                    sum_30 = sum_group_val_30[0] + sum_group_val_30[1] + sum_group_val_30[2] + sum_group_val_30[3];
                    sum_31 = sum_group_val_31[0] + sum_group_val_31[1] + sum_group_val_31[2] + sum_group_val_31[3];
                    sum_32 = sum_group_val_32[0] + sum_group_val_32[1] + sum_group_val_32[2] + sum_group_val_32[3];
                    sum_33 = sum_group_val_33[0] + sum_group_val_33[1] + sum_group_val_33[2] + sum_group_val_33[3];
                    c[  i   * size + j + 0] += sum_00; c[  i   * size + j + 1] += sum_01; c[  i   * size + j + 2] += sum_02; c[  i   * size + j + 3] += sum_03;
                    c[(i+1) * size + j + 0] += sum_10; c[(i+1) * size + j + 1] += sum_11; c[(i+1) * size + j + 2] += sum_12; c[(i+1) * size + j + 3] += sum_13;
                    c[(i+2) * size + j + 0] += sum_20; c[(i+2) * size + j + 1] += sum_21; c[(i+2) * size + j + 2] += sum_22; c[(i+2) * size + j + 3] += sum_23;
                    c[(i+3) * size + j + 0] += sum_30; c[(i+3) * size + j + 1] += sum_31; c[(i+3) * size + j + 2] += sum_32; c[(i+3) * size + j + 3] += sum_33;
                    b_transport_addr = (int32x4_t *)((int *)b_transport_addr + 4 * size);
                }
                a_addr = (int32x4_t *)((int *)a_addr + 4 * size);
            }
        }
    }
}

void Gemm(const int &size) {
    memset(&c, 0, 4194304);
    std::thread workers[num_threads];
    for(int i = 1; i < num_threads; ++i){
        workers[i] = std::thread(transpose, size, i);
    }
    transpose(size, 0);
    for(int i = 1; i < num_threads; ++i){
        workers[i].join();
    }
    for(int i = 1; i < num_threads; ++i){
        workers[i] = std::thread(multi_threads_gemm, size, i);
    }
    multi_threads_gemm(size, 0);
    for(int i = 1; i < num_threads; ++i){
        workers[i].join();
    }
}

void CheckResult(const int &size, const string &result_path) {
    ifstream file_result(result_path);
    int nelems = size * size;
    float res_i;
    for(int i = 0; i < nelems; i++) {
        file_result >> res_i;
        if(c[i] != res_i){
            printf("idx:%d, %d : %f\n", i, c[i], res_i);
            assert(c[i] == res_i);
        }
    }
    file_result.close();
}

// c = a * b
void Benchmark(const int &size) {
    const int nelems = size * size;
    const string a_path(data_path+to_string(size)+"/a");
    const string b_path(data_path+to_string(size)+"/b");
    const string result_path(data_path+to_string(size)+"/result");
    ifstream file_a(a_path);
    ifstream file_b(b_path);

    for(int i = 0; i < nelems; i++) {
        file_a >> a[i];
    }
    for(int i = 0; i < nelems; i++) {
        file_b >> b[i];
    }

    PRINT_TIME(
        Gemm(size);
    );
    
    CheckResult(size, result_path);

    file_a.close();
    file_b.close();
}

int main() {
    for(auto size: scale) {
        cout << "Running, dataset: size " << size << endl;
        Benchmark(size);
        cout << "Passed, dataset: size " << size << endl;
        cout << endl;
    }
    return 0;
}