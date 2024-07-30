#include "include/errorCheckUtils.cuh"
#include "include/kernels.cuh"

#include <cuda_runtime.h>

#include <random>
#include <iostream>


#define ceil(a, b) ( a + b - 1 )/b // utility function to calculate ceil(a/b) 

float randomFloat() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float randomInt() {
	return rand();
}


template<typename TP>
inline void initArray(TP* A, int size) {
	for (int i = 0; i < size; ++i)
		if (std::is_same<TP, float>::value) {
			A[i] = randomFloat();
		}
		else if(std::is_same<TP, int>::value) {
			A[i] = randomInt();
		}
}

template<typename TP>
inline bool checkResult(TP *A, TP *A_T_d, int M, int N) {
    for(int i = 0; i< M; ++i){
        for(int j= 0; j< N; ++j){
            if(A[i * N + j] != A_T_d[j * M + i]) return false;
        }
    }
    return true;
}

// checks if A == B
template<typename TP>
inline bool checkCopy(TP *A, TP *B, int M, int N){
    for(int i= 0; i< M*N; ++i){
        if(A[i] != B[i]) return false;
    }
    return true;
}

// checks if A = AT
template<typename TP>
inline bool checkTransposed(TP *A, TP *AT, int M, int N){
    for(int r = 0; r< M; ++r){
        for(int c =  0; c< N; ++c){
            if(A[r * N + c]!= AT[c* M + r]) return false;
        }
    }
    return true;
}

template<typename TP>
TP* callKernelCopy(TP *A, int M, int N){
    // org array
    TP *A_d;
    CUDA_CALL( cudaMalloc((void **)&A_d, M * N * sizeof(TP)) );
    CUDA_CALL( cudaMemcpy(A_d, A, M * N * sizeof(TP), cudaMemcpyHostToDevice) );

    // copy array
    TP *B_d;
    CUDA_CALL( cudaMalloc((void **)&B_d, M * N * sizeof(TP)) );

    const int TILE_WIDTH = 32;
    const int ROW_BLOCK = 8;
    dim3 block(TILE_WIDTH, ROW_BLOCK);
    dim3 grid(ceil(N, TILE_WIDTH), ceil(M, TILE_WIDTH) );

    kernelCopy<TP, TILE_WIDTH, ROW_BLOCK><<<grid, block>>>(A_d, B_d, M, N);
    cudaDeviceSynchronize();

    TP *B_h = (TP *)malloc(M * N * sizeof(TP));
    CUDA_CALL( cudaMemcpy(B_h, B_d, M * N * sizeof(TP), cudaMemcpyDeviceToHost) );
    return B_h;
}

template<typename TP>
TP* callKernelTranspose1(TP *A, int M, int N){
    // org array
    TP *A_d;
    CUDA_CALL( cudaMalloc((void **)&A_d, M * N * sizeof(TP)) );
    CUDA_CALL( cudaMemcpy(A_d, A, M * N * sizeof(TP), cudaMemcpyHostToDevice) );

    // transposed  array
    TP *A_T_d;
    CUDA_CALL( cudaMalloc((void **)&A_T_d, M * N * sizeof(TP)) );

    const int TILE_WIDTH = 32;
    const int ROW_BLOCK = 8;
    dim3 block(TILE_WIDTH, ROW_BLOCK);
    dim3 grid(ceil(N, TILE_WIDTH), ceil(M, TILE_WIDTH) );


    kernelTranspose1<TP, TILE_WIDTH, ROW_BLOCK><<<grid, block>>>(A_d, A_T_d, M, N);
    cudaDeviceSynchronize();

    TP *A_T_h = (TP *)malloc(M * N * sizeof(TP));
    CUDA_CALL( cudaMemcpy(A_T_h, A_T_d, M * N * sizeof(TP), cudaMemcpyDeviceToHost) );
    return A_T_h;
}


template<typename TP>
TP* callKernelTranspose2(TP *A, int M, int N){
    // org array
    TP *A_d;
    CUDA_CALL( cudaMalloc((void **)&A_d, M * N * sizeof(TP)) );
    CUDA_CALL( cudaMemcpy(A_d, A, M * N * sizeof(TP), cudaMemcpyHostToDevice) );

    // transposed
    TP *A_T_d;
    CUDA_CALL( cudaMalloc((void **)&A_T_d, M * N * sizeof(TP)) );

    const int TILE_WIDTH = 32;
    const int ROW_BLOCK = 8;
    dim3 block(TILE_WIDTH, ROW_BLOCK);
    dim3 grid(ceil(N, TILE_WIDTH), ceil(M, TILE_WIDTH) );


    kernelTranspose2<TP, TILE_WIDTH, ROW_BLOCK><<<grid, block>>>(A_d, A_T_d, M, N);
    cudaDeviceSynchronize();

    TP *A_T_h = (TP *)malloc(M * N * sizeof(TP));
    CUDA_CALL( cudaMemcpy(A_T_h, A_T_d, M * N * sizeof(TP), cudaMemcpyDeviceToHost) );
    return A_T_h;
}


template<typename TP>
TP* callKernelTranspose3(TP *A, int M, int N){
    // org array
    TP *A_d;
    CUDA_CALL( cudaMalloc((void **)&A_d, M * N * sizeof(TP)) );
    CUDA_CALL( cudaMemcpy(A_d, A, M * N * sizeof(TP), cudaMemcpyHostToDevice) );

    // copy array
    TP *A_T_d;
    CUDA_CALL( cudaMalloc((void **)&A_T_d, M * N * sizeof(TP)) );

    const int TILE_WIDTH = 32;
    const int ROW_BLOCK = 8;
    dim3 block(TILE_WIDTH, ROW_BLOCK);
    dim3 grid(ceil(N, TILE_WIDTH), ceil(M, TILE_WIDTH) );


    kernelTranspose3<TP, TILE_WIDTH, ROW_BLOCK><<<grid, block>>>(A_d, A_T_d, M, N);
    cudaDeviceSynchronize();

    TP *A_T_h = (TP *)malloc(M * N * sizeof(TP));
    CUDA_CALL( cudaMemcpy(A_T_h, A_T_d, M * N * sizeof(TP), cudaMemcpyDeviceToHost) );
    return A_T_h;
}




int main(){
    int M, N;
    M = 4096, N = 4096;
    #define DTYPE int
    DTYPE *A= (DTYPE*)malloc(M * N * sizeof(DTYPE));
    initArray<DTYPE>(A, M * N);

    DTYPE *A_copy = callKernelCopy<DTYPE>(A, M, N);
    std::cout<<"\n IS COPY CORRECT: "<<checkCopy<DTYPE>(A, A_copy, M, N);

    DTYPE *A_T = callKernelTranspose1<DTYPE>(A, M, N);
    std::cout<<"\n IS TRANSPOSE 1 CORRECT: "<<checkTransposed<DTYPE>(A, A_T, M, N);

    A_T = callKernelTranspose2<DTYPE>(A, M, N);
    std::cout<<"\n IS TRANSPOSE 2 CORRECT: "<<checkTransposed<DTYPE>(A, A_T, M, N);

    A_T = callKernelTranspose3<DTYPE>(A, M, N);
    std::cout<<"\n IS TRANSPOSE 3 CORRECT: "<<checkTransposed<DTYPE>(A, A_T, M, N);

    

    return 0;
}