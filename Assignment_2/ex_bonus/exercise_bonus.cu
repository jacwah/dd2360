#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>

#define SEED     9291
#define NUM_ITER 1e9

#ifndef PRECISION
#define PRECISION 2
#endif

#if PRECISION == 1
    #define real float
#else
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
        #if PRECISION != 1
            x = curand_uniform_double(&state[threadIdx.x]);
            y = curand_uniform_double(&state[threadIdx.x]);
        #else 
            x = curand_uniform(&state[threadIdx.x]);
            y = curand_uniform(&state[threadIdx.x]);
        #endif
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
    unsigned long long *counts = NULL;
    cudaMalloc(&counts, sizeof(unsigned long long)*48); //48 is the largest amount of blocks that will be used
    
    unsigned long long *counts_h = (unsigned long long*)malloc(sizeof(unsigned long long)*48);
    
    for(int i = 16; i <= 256; i*=2) {
        unsigned long long int count = 0;
        timeval t1, t2;
        
        int blocks = ((3*256) + (i - 1))/i;
        printf("Blocks: %i  Threads per block: %i\n", blocks, i);
        //printf("%i, %i\n", blocks, i);
        
        gettimeofday(&t1, NULL);
   
        cudaMemset(counts, 0, sizeof(unsigned long long)*blocks);    
        
        long long iterations = (NUM_ITER + (i*blocks - 1))/ (i*blocks);
        printf("Total iterations: %lld\n", iterations*blocks*i);
        //printf("%lld\n", iterations*blocks*i);
        
        calc_prob<<<blocks, i, i*sizeof(curandState)>>>(iterations, counts);
        
        cudaMemcpy(counts_h, counts, sizeof(unsigned long long) * blocks, cudaMemcpyDefault);
        
        for (int j = 0; j < blocks; j++) {
            // printf("Counts_h[%d] = %llu\n", j, counts_h[j]);
            count += counts_h[j];
        }
        
        // Estimate Pi and display the result
        double pi = ((double)count / (double)(iterations*i*blocks)) * 4.0;
        gettimeofday(&t2, NULL);
        printf("The error is %e, time: %e\n", (abs(M_PI - pi)), (t2.tv_sec + t2.tv_usec/1e6) - (t1.tv_sec + t1.tv_usec/1e6));
        //printf("%e, %e\n", (abs(M_PI - pi)), (t2.tv_sec + t2.tv_usec/1e6) - (t1.tv_sec + t1.tv_usec/1e6));

    }
    
    cudaFree(counts);
    free(counts_h);   
    
    return 0;
}
