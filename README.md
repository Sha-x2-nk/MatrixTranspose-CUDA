# CUDA Transpose Kernels
## Introduction
This project implements three transpose kernels using CUDA, starting from a basic implementation and gradually optimizing for improved performance. Transposing matrices efficiently is crucial for many computational tasks, especially in deep learning and scientific computing.

## Kernels and Performance

#### HARDWARE: RTX 3070Ti ( Compute Capablity 8.6 )

The benchmark was performed on matrix of size 4096 x 4096 (although the code works for any arbitrary size) using Nvidia Nsight Compute.

|Kernels|Runtime| % of copy speed|
|--|--|--|
|1. Copy Kernel | 353.3 us | 100 % |
|2. Transpose Naive | 1.63 ms | 21 % |
|3. Transpose in Shared Memory | 508 us | 69 %|
|4. Resolving Bank Conflicts | 361.3 us | 97.7 % |

After our final optimisation, we are transposing at ~98% of copy speed.

## Usage
* Compile using nvcc

    <code>nvcc main.cu -o main.exe</code>

* Run

    <code>main.exe</code>

* Tune parameters like BLOCK_SIZE for your hardware.

## Acknowledgements
Mark Harris' amazing article on optimising transpose in CUDA
https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/