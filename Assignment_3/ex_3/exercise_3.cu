#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>


// 40 GB memory
// sizeof(Particle) = 4*3*2 = 24 bytes
// 40 GB / 24 B = 1.6 G particles
#define NUM_PARTICLES 1e5
#define NUM_ITERATIONS 10


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

__global__ void update_kernel(Particle* particles, const uint64_t n) { 
    uint64_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {    
        update_particle(particles+i);
    }
}

void update(Particle* particles, const uint64_t n) {
#pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
        update_particle(particles+i);
    }
}

    
void initialize_array(Particle *particles, const uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        particles[i].position.x = (float)rand()/RAND_MAX; 
        particles[i].position.y = (float)rand()/RAND_MAX; 
        particles[i].position.z = (float)rand()/RAND_MAX; 
        particles[i].velocity = {}; 
    }
}

void compare(const Particle *p_cpu, const Particle *p_gpu, const uint64_t n) {
#pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
        if (abs(p_cpu[i].position.x - p_gpu[i].position.x) > 1e6 ||
        abs(p_cpu[i].position.y - p_gpu[i].position.y) > 1e6 || 
        abs(p_cpu[i].position.z - p_gpu[i].position.z) > 1e6 ||
        abs(p_cpu[i].velocity.x - p_gpu[i].velocity.x) > 1e6 ||
        abs(p_cpu[i].velocity.y - p_gpu[i].velocity.y) > 1e6 ||
        abs(p_cpu[i].velocity.z - p_gpu[i].velocity.z) > 1e6) {
            printf("Result not equal\n");
        }
    }
    printf("Comparison done\n");
}


int main() {
    Particle *particles_d = NULL;
    uint64_t arraySize = sizeof(Particle)*NUM_PARTICLES;    
    timeval gpu_t1;   
    timeval gpu_t2;   
    Particle *particles = NULL;
    //Particle *particles_cpu = (Particle*)malloc(arraySize);

    const uint64_t BLOCK_SIZE = 64;
    const uint64_t NUM_BATCHES = 4*1e3;
    const uint64_t BATCH_SIZE = NUM_PARTICLES / NUM_BATCHES;
    const uint64_t NUM_STREAMS = 8;
    const uint64_t BATCHES_PER_STREAM = NUM_BATCHES / NUM_STREAMS;
    
    cudaMallocHost(&particles, arraySize);
    cudaMalloc(&particles_d, arraySize);

    initialize_array(particles, NUM_PARTICLES);
    
    /*
    memcpy(particles_cpu, particles, arraySize);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        update(particles_cpu, NUM_PARTICLES);
    }
    */

    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);
    
    gettimeofday(&gpu_t1, NULL);

    for (uint64_t i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < NUM_STREAMS; j++) {
            cudaStream_t stream = streams[j];

            for (int k = 0; k < BATCHES_PER_STREAM; k++) {
                uint64_t offset = j*BATCHES_PER_STREAM*BATCH_SIZE + k*BATCH_SIZE;

                Particle *particle_batch = &particles[offset];
                Particle *particle_batch_d = &particles_d[offset];

                cudaMemcpyAsync(particle_batch_d, particle_batch, BATCH_SIZE*sizeof(Particle), cudaMemcpyDefault, stream);
                update_kernel<<<(BATCH_SIZE + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(particle_batch_d, BATCH_SIZE);
                cudaMemcpyAsync(particle_batch, particle_batch_d, BATCH_SIZE*sizeof(Particle), cudaMemcpyDefault, stream);
            }
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&gpu_t2, NULL);
    printf("GPU time: %e seconds \n", ((gpu_t2.tv_sec + gpu_t2.tv_usec/1e6) - (gpu_t1.tv_sec + gpu_t1.tv_usec/1e6)));
    
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);

    //compare(particles_cpu, particles, NUM_PARTICLES);
    
    cudaFreeHost(particles);
    cudaFree(particles_d);
}
