#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <thread>
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


void transpose(const int size, const vec input, vec &result, 
                int thread_id){
    int single_processing_line = size / num_threads;
    int start = single_processing_line * thread_id;
    int end = single_processing_line * (thread_id + 1);
    for(int i = start; i < end; ++i){
        for(int j = 0; j < size; ++j){
            result[i*size + j] = input[j*size + i];
        }
    }
}

void multi_threads_gemm(const int size, const vec a, const vec b, 
                vec &c, int thread_id){                
    int single_processing_line = size / num_threads; 
    int start = single_processing_line * thread_id;
    int end = single_processing_line * (thread_id + 1);
    for(int i = start; i < end; ++i){
        for(int j = 0; j < size; ++j){
            int sum;
            int32x4_t sum_vec;
            int32x4x4_t sum_group;
            int32x4x4_t a_vec, b_vec;
            sum_group.val[0] = {0, 0, 0, 0};
            sum_group.val[1] = {0, 0, 0, 0};
            sum_group.val[2] = {0, 0, 0, 0};
            sum_group.val[3] = {0, 0, 0, 0};
            for(int k = 0; k < size; k += 16){
                a_vec = vld1q_s32_x4((int *)&a[i * size + k]);
                b_vec = vld1q_s32_x4((int *)&b[j * size + k]);
                sum_group.val[0] = vmlaq_s32(sum_group.val[0], a_vec.val[0], b_vec.val[0]);
                sum_group.val[1] = vmlaq_s32(sum_group.val[1], a_vec.val[1], b_vec.val[1]);
                sum_group.val[2] = vmlaq_s32(sum_group.val[2], a_vec.val[2], b_vec.val[2]);
                sum_group.val[3] = vmlaq_s32(sum_group.val[3], a_vec.val[3], b_vec.val[3]);
            }
            sum_vec = vaddq_s32(vaddq_s32(sum_group.val[0], sum_group.val[1]), vaddq_s32(sum_group.val[2], sum_group.val[3]));
            sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
            c[i * size + j] = sum;
        }
    }
}

void Gemm(const int &size, vec &a, vec &b, vec &c) {
    vec b_transposed(size*size, 0);
    std::thread workers[num_threads];
    for(int i = 1; i < num_threads; ++i){
        workers[i] = std::thread(transpose, size, b, 
                                std::ref(b_transposed), i);
    }
    transpose(size, b, b_transposed, 0);
    for(int i = 1; i < num_threads; ++i){
        workers[i].join();
    }
    for(int i = 1; i < num_threads; ++i){
        workers[i] = std::thread(multi_threads_gemm, size, a, 
                                b_transposed, std::ref(c), i);
    }
    multi_threads_gemm(size, a, b_transposed, c, 0);
    for(int i = 1; i < num_threads; ++i) {
        workers[i].join();
    }
}

void CheckResult(const vec &c, const string &result_path) {
    ifstream file_result(result_path);
    int nelems = c.size();
    float res_i;
    for(int i = 0; i < nelems; i++) {
        file_result >> res_i;
        assert(c[i] == res_i);
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

    vec a(nelems, 0);
    vec b(nelems, 0);
    vec c(nelems, 0);

    for(int i = 0; i < nelems; i++) {
        file_a >> a[i];
    }
    for(int i = 0; i < nelems; i++) {
        file_b >> b[i];
    }

    PRINT_TIME(
        Gemm(size, a, b, c);
    );
    
    CheckResult(c, result_path);

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