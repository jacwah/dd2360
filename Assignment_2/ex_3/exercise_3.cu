#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define NUM_PARTICLES 10000000
#define NUM_ITERATIONS 10000

struct Particle {
    float3 position;
    float3 velocity;
};

__host__ __device__ void update_particle(Particle* particle) { 
        particle->velocity.x = particle->position.x;
        particle->velocity.y = particle->position.y;
        particle->velocity.z = particle->position.z;
        particle->position.x += particle->velocity.x; 
        particle->position.y += particle->velocity.y;
        particle->position.z += particle->velocity.z;
}

__global__ void update_kernel(Particle* particles, const int n) { 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {    
        update_particle(particles+i);
    }
}

void update(Particle* particles, const int n) {
    for (int i = 0; i < n; i++) {
        update_particle(particles+i);
    }
}

    
void initialize_array(Particle *particles, const int n) {
    for (int i = 0; i < n; i++) {
        particles[i].position.x = (float)rand()/RAND_MAX; 
        particles[i].position.y = (float)rand()/RAND_MAX; 
        particles[i].position.z = (float)rand()/RAND_MAX; 
        particles[i].velocity = {}; 
    }
}

void compare(const Particle *p_cpu, const Particle *p_gpu, const int n) {
    for (int i = 0; i < n; i++) {
        if (abs(p_cpu[i].position.x - p_gpu[i].position.x) > 1e6 ||
        abs(p_cpu[i].position.y - p_gpu[i].position.y) > 1e6 || 
        abs(p_cpu[i].position.z - p_gpu[i].position.z) > 1e6 ||
        abs(p_cpu[i].velocity.x - p_gpu[i].velocity.x) > 1e6 ||
        abs(p_cpu[i].velocity.y - p_gpu[i].velocity.y) > 1e6 ||
        abs(p_cpu[i].velocity.z - p_gpu[i].velocity.z) > 1e6) {
            printf("Result not equal\n");
            return;
        }
    }
    printf("Comparison OK\n");
}


int main() {
    Particle *particles_d = NULL;
    int arraySize = sizeof(Particle)*NUM_PARTICLES;    
    Particle *particles = (Particle*)malloc(arraySize);
    Particle *particles_cp = (Particle*)malloc(arraySize);
    Particle *particles_i = (Particle*)malloc(arraySize);
    timeval *gpu_t1 = (timeval*) malloc(sizeof(timeval));   
    timeval *gpu_t2 = (timeval*) malloc(sizeof(timeval));   
    timeval *cpu_t1 = (timeval*) malloc(sizeof(timeval));   
    timeval *cpu_t2 = (timeval*) malloc(sizeof(timeval));   

    initialize_array(particles, NUM_PARTICLES);
    
    cudaMalloc(&particles_d, arraySize);
    memcpy(particles_i, particles, arraySize);
    
    printf("Computing simulation on the CPU...\n");
    gettimeofday(cpu_t1, NULL);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        update(particles, NUM_PARTICLES);
    }
    gettimeofday(cpu_t2, NULL);
    printf("CPU time: %e seconds \n", ((cpu_t2->tv_sec + cpu_t2->tv_usec/1e6) - (cpu_t1->tv_sec + cpu_t1->tv_usec/1e6)));
    
    printf("Computing simulation on the GPU...\n");
    
    for (int j = 16; j <= 256; j*=2) {
        printf("Block size: %d\n", j);
        gettimeofday(gpu_t1, NULL);
        cudaMemcpy(particles_d, particles_i, arraySize, cudaMemcpyDefault);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            update_kernel<<<(NUM_PARTICLES + j-1)/j,j>>>(particles_d, NUM_PARTICLES);
        }
        cudaMemcpy(particles_cp, particles_d, arraySize, cudaMemcpyDefault);
        gettimeofday(gpu_t2, NULL);

        compare(particles, particles_cp, NUM_PARTICLES); 
        printf("GPU time: %e seconds \n", ((gpu_t2->tv_sec + gpu_t2->tv_usec/1e6) - (gpu_t1->tv_sec + gpu_t1->tv_usec/1e6)));
    }
   
    
    free(particles);
    free(particles_i);
    free(particles_cp);
    cudaFree(particles_d);
    free(gpu_t1);
    free(gpu_t2);
    free(cpu_t1);
    free(cpu_t2);
    return 0;
}
