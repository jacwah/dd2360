#include <stdio.h>
//#define N 64
//#define TPB 32

__global__ void hello_kernel() {
    int id = threadIdx.x;    

    printf("Hello World! My ThreadId is %d \n", id);
}

int main() {
    
    hello_kernel<<<1,256>>>();    
    cudaDeviceSynchronize();
    return 0;
}
