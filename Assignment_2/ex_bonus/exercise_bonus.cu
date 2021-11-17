#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define SEED     921
#define NUM_ITER 1000000000 

__global__ void calc_prob(const int iterations, unsigned long long *counts) {
    unsigned long long count = 0;
    double x, y, z;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    extern __shared__ curandState state[];
    
    curand_init(SEED, idx, 0, &state[threadIdx.x]); 
    for (int iter = 0; iter < iterations; iter++)
    {
        // Generate random (X,Y) points
        x = curand_uniform(&state[threadIdx.x]);
        y = curand_uniform(&state[threadIdx.x]);
        //printf("x = %f, y = %f\n", x, y);
        z = sqrt((x*x) + (y*y));
        // printf("z=%f\n", z);  
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }
    printf("Thread idx: %d count: %llu\n",idx,count);
    atomicAdd_block(&counts[blockIdx.x], count);
}


int main(int argc, char* argv[])
{
    //double pi;
    //int blocks, iterations;
    //srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    for(int i = 16; i <= 256; i*=2) {
        unsigned long long int count = 0;
        unsigned long long *counts = NULL;
        //curandState *dev_random = NULL;
        
        int blocks = (640 + (i - 1))/i;
        
        cudaMalloc(&counts, sizeof(unsigned long long)*blocks);
        cudaMemset(counts, 0, sizeof(unsigned long long)*blocks);
        unsigned long long *counts_h = (unsigned long long*)malloc(sizeof(unsigned long long)*blocks);
        
        //cudaMalloc(&dev_random, blocks*i*sizeof(curandState));
        
        int iterations = (NUM_ITER + (i*blocks - 1))/ i*blocks;
        
        calc_prob<<<blocks, i, i*sizeof(curandState)>>>(iterations, counts);
        
        cudaMemcpy(counts_h, counts, sizeof(unsigned long long) * blocks, cudaMemcpyDefault);
        
        for (int j = 0; j < blocks; j++) {
            printf("Counts_h[%d] = %llu\n", j, counts_h[j]);
            count += counts_h[j];
        }
        
        // Estimate Pi and display the result
        double pi = ((double)count / (double)(iterations*i*blocks)) * 4.0;
    
        printf("The result is %f\n", pi);
        
    }

    //cudaFree(counts);
    //cudaFree(dev_random);
    //free(counts_h);
    
    
    return 0;
}

