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

int a[4194304] __attribute__((__aligned__((32))));
int b[4194304] __attribute__((__aligned__((32))));
int c[4194304] __attribute__((__aligned__((32))));
int b_transport[4194304] __attribute__((__aligned__((32))));

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
    int byte_a_line = 4 * size;
    int sum;
    int32x4_t sum_vec;
    int32x4_t sum_group_val_0, sum_group_val_1, sum_group_val_2, sum_group_val_3;
    int32x4_t a_vec_val_0, a_vec_val_1, a_vec_val_2, a_vec_val_3;
    int32x4_t b_vec_val_0, b_vec_val_1, b_vec_val_2, b_vec_val_3;
    int32x4_t * a_addr = (int32x4_t *)((char *)&a + start * byte_a_line);
    int32x4_t * a_group_addr_now;
    int32x4_t * b_group_addr_now;
    do {
        int j = 0;
        int32x4_t * b_transport_addr = (int32x4_t *)&b_transport;
        do {
            int k = 0;
            sum_group_val_0 = {0, 0, 0, 0};
            sum_group_val_1 = {0, 0, 0, 0};
            sum_group_val_2 = {0, 0, 0, 0};
            sum_group_val_3 = {0, 0, 0, 0};
            a_group_addr_now = a_addr;
            b_group_addr_now = b_transport_addr;
            do {
                a_vec_val_0 = a_group_addr_now[0];
                a_vec_val_1 = a_group_addr_now[1];
                a_vec_val_2 = a_group_addr_now[2];
                a_vec_val_3 = a_group_addr_now[3];
                a_group_addr_now += 4;
                b_vec_val_0 = b_group_addr_now[0];
                b_vec_val_1 = b_group_addr_now[1];
                b_vec_val_2 = b_group_addr_now[2];
                b_vec_val_3 = b_group_addr_now[3];
                b_group_addr_now += 4;
                sum_group_val_0 = vmlaq_s32(sum_group_val_0, b_vec_val_0, a_vec_val_0);
                sum_group_val_1 = vmlaq_s32(sum_group_val_1, b_vec_val_1, a_vec_val_1);
                sum_group_val_2 = vmlaq_s32(sum_group_val_2, b_vec_val_2, a_vec_val_2);
                sum_group_val_3 = vmlaq_s32(sum_group_val_3, b_vec_val_3, a_vec_val_3);
                k += 16;
            } while(k < size);
            sum_vec = vaddq_s32(vaddq_s32(vaddq_s32(sum_group_val_0, sum_group_val_1), sum_group_val_2), sum_group_val_3);
            sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
            c[i * size + j] = sum;
            b_transport_addr = (int32x4_t *)((char *)b_transport_addr + byte_a_line);
            ++j;
        } while(j != size);
        ++i;
        a_addr = (int32x4_t *)((char *)a_addr + byte_a_line);
    } while(i != end);
}

void Gemm(const int &size) {
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