#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <CL/cl.h>
#include <string.h>


// 40 GB memory
// sizeof(Particle) = 4*3*2 = 24 bytes
// 40 GB / 24 B = 1.6 G particles
#define NUM_PARTICLES 1e6
#define NUM_ITERATIONS 10

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));
// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

struct float3 {
    float x, y, z;
};

struct Particle {
    float3 position;
    float3 velocity;
};


void update_particle(Particle* particle) {
        particle->velocity.x = particle->position.x;
        particle->velocity.y = particle->position.y;
        particle->velocity.z = particle->position.z;
        particle->position.x += particle->velocity.x;
        particle->position.y += particle->velocity.y;
        particle->position.z += particle->velocity.z;
};

const char* update_kernel = 
"struct float3 {"
"    float x, y, z;"
"};"
"struct Particle {"
"    struct float3 position;"
"    struct float3 velocity;"
"};"
"void update_particle(__global struct Particle* particle) {" 
"        particle->velocity.x = particle->position.x;"
"        particle->velocity.y = particle->position.y;"
"        particle->velocity.z = particle->position.z;"
"        particle->position.x += particle->velocity.x;"
"        particle->position.y += particle->velocity.y;"
"        particle->position.z += particle->velocity.z;"
"}"
"__kernel void update_kernel(__global struct Particle* particles, const ulong n) { "
"    ulong i = get_global_id(0);"
"    if (i < n) {  "  
"        update_particle(particles+i);"
"    }"
"}";

void update(Particle* particles, const uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        update_particle(particles+i);
    }
}

    
void initialize_array(Particle *particles, const uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        particles[i].position.x = rand()/(float)RAND_MAX; 
        particles[i].position.y = rand()/(float)RAND_MAX; 
        particles[i].position.z = rand()/(float)RAND_MAX; 
        particles[i].velocity = {}; 
    }
}

void compare(const Particle *p_cpu, const Particle *p_gpu, const uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        if (abs(p_cpu[i].position.x - p_gpu[i].position.x) > 1e-6 ||
        abs(p_cpu[i].position.y - p_gpu[i].position.y) > 1e-6 || 
        abs(p_cpu[i].position.z - p_gpu[i].position.z) > 1e-6 ||
        abs(p_cpu[i].velocity.x - p_gpu[i].velocity.x) > 1e-6 ||
        abs(p_cpu[i].velocity.y - p_gpu[i].velocity.y) > 1e-6 ||
        abs(p_cpu[i].velocity.z - p_gpu[i].velocity.z) > 1e-6) {
            printf("Result not equal\n");
            return;
        }
    }
    printf("Comparison OK\n");
}

int main() {
    uint64_t arraySize = sizeof(Particle)*NUM_PARTICLES;  
    unsigned long n_particles = NUM_PARTICLES;  
    timeval gpu_t1;   
    timeval gpu_t2;
    timeval cpu_t1;   
    timeval cpu_t2;
    // const uint64_t BLOCK_SIZE = 64;
    Particle *particles = (Particle*)malloc(arraySize);
    Particle *particles_copy = (Particle*)malloc(arraySize);

    cl_platform_id * platforms; cl_uint n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

    // Find and sort devices
    cl_device_id *device_list; cl_uint n_devices;
    err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
    CHK_ERROR(err);
    
    // Create and initialize an OpenCL context
    cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);
    CHK_ERROR(err);

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueueWithProperties(context, device_list[0], 0, &err);
    CHK_ERROR(err);

    initialize_array(particles, NUM_PARTICLES);
    memcpy(particles_copy, particles, arraySize);
    
    //char *sources[] = {};
    //cl_program prog = clCreateProgramWithSource(context, sizeof(sources)/sizeof(sources[0]), sources, NULL, &err);
    cl_program prog = clCreateProgramWithSource(context, 1, &update_kernel, NULL, &err);
    CHK_ERROR(err);
    err = clBuildProgram(prog, 1, device_list, NULL, NULL, NULL);
    CHK_ERROR(err);

    if (err != CL_SUCCESS) { 
        size_t len = 0;
        cl_int ret = CL_SUCCESS;
        ret = clGetProgramBuildInfo(prog, device_list[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char *buffer = (char*)calloc(len, sizeof(char));
        ret = clGetProgramBuildInfo(prog, device_list[0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

        // char buffer[2048] = {};
        // clGetProgramBuildInfo(prog, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        
        fprintf(stderr,"Build error: %s\n", buffer);
        return 1; 
        }

    cl_mem particles_d = clCreateBuffer(context, CL_MEM_READ_WRITE, arraySize, NULL, &err);
    CHK_ERROR(err);

    cl_kernel kernel = clCreateKernel(prog, "update_kernel", &err); CHK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &particles_d); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(unsigned long), (void *) &n_particles); CHK_ERROR(err);

    for (int blockSize = 8; blockSize <= 256; blockSize *= 2) {
        gettimeofday(&gpu_t1, NULL);
            
        clEnqueueWriteBuffer(cmd_queue, particles_d, CL_TRUE, 0, arraySize, particles_copy, 0, NULL, NULL);
        CHK_ERROR(err);

        for (uint64_t i = 0; i < NUM_ITERATIONS; i++) {
            size_t n_workitem = ((((size_t)NUM_PARTICLES) + blockSize-1)/blockSize) * blockSize;
            size_t workgroup_size = blockSize;
            err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL);
            CHK_ERROR(err);
        }

        clEnqueueReadBuffer(cmd_queue, particles_d, CL_TRUE, 0, arraySize, particles, 0, NULL, NULL);
        CHK_ERROR(err);

        gettimeofday(&gpu_t2, NULL);
        printf("GPU time: %e seconds, block size %d \n", ((gpu_t2.tv_sec + gpu_t2.tv_usec/1e6) - (gpu_t1.tv_sec + gpu_t1.tv_usec/1e6)), blockSize);
    }
    
     // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
    
    // clReleaseContext frees memory allocated in the context, as well as command queues etc
    err = clReleaseContext(context); CHK_ERROR(err);
    free(platforms);
    free(device_list);

    // Run simulation on CPU:
    gettimeofday(&cpu_t1, NULL);
    for (uint64_t i = 0; i < NUM_ITERATIONS; i++) {
        update(particles_copy, n_particles);
    }
    gettimeofday(&cpu_t2, NULL);
    printf("CPU time: %e seconds \n", ((cpu_t2.tv_sec + cpu_t2.tv_usec/1e6) - (cpu_t1.tv_sec + cpu_t1.tv_usec/1e6)));

    compare(particles_copy, particles, NUM_PARTICLES);

    free(particles);
    free(particles_copy);
    //cudaFreeHost(particles);
    //cudaFree(particles_d);
}

const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";  
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}
