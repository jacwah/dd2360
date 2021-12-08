#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>

#define ARRAY_SIZE 1e8

void compare(float *x, float *y, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float diff = abs(x[i] - y[i]);
    if (diff > 1e-6) {
      printf("Arrays are not equal! Index: %ld, Values: %e, %e, diff %e\n", i, x[i], y[i], diff);
      return;
    }
  }
  printf("OK!\n");
}

void saxpy_cpu(float* x, float* y, const float a) {
    for (size_t i = 0; i < ARRAY_SIZE; i ++) {
        y[i] += a*x[i];    
    }
}

void saxpy_gpu(float* restrict x, float* restrict y, const float a) {
    size_t n = ARRAY_SIZE;
    #pragma acc parallel loop copyin(x[0:n]) copy(y[0:n])
        for (size_t i = 0; i < n; i ++) {
            y[i] += a*x[i];
        }
}

void initialize_array(float *x, float *y, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        x[i] = rand()/(float)RAND_MAX; 
        y[i] = rand()/(float)RAND_MAX;
    }
}

int main() {
    int arraySize = sizeof(float)*ARRAY_SIZE;    
    float *x = (float*)malloc(arraySize);
    float *y = (float*)malloc(arraySize);
    float *d_y = (float*)malloc(arraySize);
    struct timeval gpu_t1;
    struct timeval gpu_t2;
    struct timeval cpu_t1;
    struct timeval cpu_t2;

    initialize_array(x, y, ARRAY_SIZE);
    memcpy(d_y, y, arraySize);
    
    printf("Computing SAXPY on the GPU...\n");
    gettimeofday(&gpu_t1, NULL);
    saxpy_gpu(x, d_y, 5);
    gettimeofday(&gpu_t2, NULL);
    printf("GPU time: %e seconds \n", ((gpu_t2.tv_sec + gpu_t2.tv_usec/1e6) - (gpu_t1.tv_sec + gpu_t1.tv_usec/1e6)));
    printf("computing SAXPY on the CPU...\n");
    
    gettimeofday(&cpu_t1, NULL);
    saxpy_cpu(x, y, 5);
    gettimeofday(&cpu_t2, NULL);
    printf("CPU time: %e seconds \n", ((cpu_t2.tv_sec + cpu_t2.tv_usec/1e6) - (cpu_t1.tv_sec + cpu_t1.tv_usec/1e6)));
    
    printf("Comparing the output for each implementation...\n");
    compare(y, d_y, ARRAY_SIZE);

    free(x);
    free(y);
    free(d_y);
    return 0;
}
