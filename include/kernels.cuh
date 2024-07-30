#pragma once

#include "errorCheckUtils.cuh"

#include <cuda_runtime.h>

#include <iostream>

/*
Copy Kernel
TILE_WIDTH = 32. one block will copy 32 x 32 elements.
*/
template<typename TP, int TILE_WIDTH, int ROW_BLOCK>
__global__ void kernelCopy(TP *A, TP* B, int M, int N){
    const int r = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int c = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // ROW_BLOCK = 8 (step size).
    if(c< N){
        for(int rowOffset = 0; (r< M) && (rowOffset< TILE_WIDTH); rowOffset+= ROW_BLOCK){ // blockDim.y is TILE_WIDTH / ROW_BLOCK
            B[(r + rowOffset) * N + c] = A[(r + rowOffset) * N + c];
        }
    }

}

/*
Transpose Kernel. Naive
TILE_WIDTH = 32. one block will copy 32 x 32 elements.
*/
template<typename TP, int TILE_WIDTH, int ROW_BLOCK>
__global__ void kernelTranspose1(TP *A, TP* B, int M, int N){
    int r = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int c = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // ROW_BLOCK = 8 (step size).
    if(c< N){
        for(int rowOffset = 0; (r + rowOffset< M) && (rowOffset< TILE_WIDTH); rowOffset+= ROW_BLOCK){ // blockDim.y is TILE_WIDTH / ROW_BLOCK
            B[c * M + (r + rowOffset)] = A[(r + rowOffset) * N + c];
        }
    }

}

/*
Transpose Kernel. Load in Shared memory first, tranpose in shmem and then store [Coalesced loads and stores]
TILE_WIDTH = 32. one block will copy 32 x 32 elements.
*/
template<typename TP, int TILE_DIM, int BLOCK_ROWS>
__global__ void kernelTranspose2(TP *idata, TP* odata, int M, int N){
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
    if(y + j < M && x< N)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*N + x];

  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
    if(y + j < N && x < M)
        odata[(y+j)*M + x] = tile[threadIdx.x][threadIdx.y + j];
  }

}

/*
Transpose Kernel. Resolve bank conflicts by adding one dimension. [without it, same column will be mapped to same memory bank]
TILE_WIDTH = 32. one block will copy 32 x 32 elements.
*/
template<typename TP, int TILE_DIM, int BLOCK_ROWS>
__global__ void kernelTranspose3(TP *idata, TP* odata, int M, int N){
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
    if(y + j < M && x< N)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*N + x];

  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS){
    if(y + j < N && x < M)
        odata[(y+j)*M + x] = tile[threadIdx.x][threadIdx.y + j];
  }

}