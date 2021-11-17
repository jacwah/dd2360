#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define SEED     921
#define NUM_ITER 1e10

#ifndef real
#define real double
#endif

__global__ void calc_prob(const long long iterations, unsigned long long *counts) {
    unsigned long long count = 0;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    extern __shared__ curandState state[];
    
    curand_init(SEED, idx, 0, &state[threadIdx.x]); 
    for (long long iter = 0; iter < iterations; iter++)
    {
        real x, y, z;

        // Generate random (X,Y) points
        x = curand_uniform(&state[threadIdx.x]);
        y = curand_uniform(&state[threadIdx.x]);
        z = (x*x) + (y*y);

        // Check if point is in unit circle
        if (z <= ((real)1.0)) {
            count++;
        }
    }

#if __CUDA_ARCH__ >= 600
    atomicAdd_block(&counts[blockIdx.x], count);
#else
    atomicAdd(&counts[blockIdx.x], count);
#endif
}


int main(int argc, char* argv[])
{
    //double pi;
    //int blocks, iterations;
    //srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    for(int i = 16; i <= 256; i*=2) {
        unsigned long long int count = 0;
        unsigned long long *counts = NULL;
        
        int blocks = (640 + (i - 1))/i;
        printf("Blocks: %i  Threads per block: %i\n", blocks, i);
        
        cudaMalloc(&counts, sizeof(unsigned long long)*blocks);
        cudaMemset(counts, 0, sizeof(unsigned long long)*blocks);
        unsigned long long *counts_h = (unsigned long long*)malloc(sizeof(unsigned long long)*blocks);
        
        
        long long iterations = (NUM_ITER + (i*blocks - 1))/ (i*blocks);
        printf("Total itterations: %lld\n", iterations);
        
        calc_prob<<<blocks, i, i*sizeof(curandState)>>>(iterations, counts);
        
        cudaMemcpy(counts_h, counts, sizeof(unsigned long long) * blocks, cudaMemcpyDefault);
        
        for (int j = 0; j < blocks; j++) {
            // printf("Counts_h[%d] = %llu\n", j, counts_h[j]);
            count += counts_h[j];
        }
        
        // Estimate Pi and display the result
        double pi = ((double)count / (double)(iterations*i*blocks)) * 4.0;
    
        printf("The result is %f\n", pi);

        cudaFree(counts);
        free(counts_h);   
    }
    
    
    return 0;
}
