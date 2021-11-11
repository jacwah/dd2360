#include <stdio.h>
#include <stdlib.h>
#define N 64
#define TPB 32
#define ARRAY_SIZE 10000

__global__ void saxpy_kernel(float* x, float* y, const float a) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;    
    if(i < ARRAY_SIZE) {
        y[i] += a*x[i];
    }
}

__global__ void compare(float *x, float *y) {
    __shared__ bool b; 
    if (threadIdx.x == 0) {
        b = false;
    }
    
    __syncthreads();
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;    
    if (i < ARRAY_SIZE && abs(x[i] - y[i]) > 1e-6) {
        printf("Mismatch %e, %e \n", x[i], y[i]);
        b = true;
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        if (b)
            printf("Mismatch in block %d \n", blockIdx.x);
        else 
            printf("Block %d correct\n", blockIdx.x);
    }
}

void saxpy_cpu(float* x, float* y, const float a) {
    for (int i = 0; i < ARRAY_SIZE; i ++) {
        y[i] += a*x[i];    
    }
}

void initialize_array(float *x, float *y, const int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (float)rand()/RAND_MAX; 
        y[i] = (float)rand()/RAND_MAX;
    }
}


int main() {
    float *xd, *yd = NULL;
    int arraySize = sizeof(float)*ARRAY_SIZE;    
    float *x = (float*)malloc(arraySize);
    float *y = (float*)malloc(arraySize);
        

    initialize_array(x, y, ARRAY_SIZE);
    
    cudaMalloc(&xd, arraySize);
    cudaMalloc(&yd, arraySize);
    cudaMemcpy(xd, x, arraySize, cudaMemcpyDefault);
    cudaMemcpy(yd, y, arraySize, cudaMemcpyDefault); 
    
    printf("Computing SAXPY on the GPU...\n");
    saxpy_kernel<<<(ARRAY_SIZE + 255)/256,256>>>(xd, yd, 5);
    printf("computing SAXPY on the CPU...\n");
    saxpy_cpu(x, y, 5);
    
    cudaMemcpy(xd, y, arraySize, cudaMemcpyDefault);
    
    printf("Comparing the output for each implementation...\n");
    compare<<<(ARRAY_SIZE + 255)/256,256>>>(xd, yd);
    cudaDeviceSynchronize();
    return 0;
}
