#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Error fetching device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Total shared memory per block: " 
                  << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Threads per warp: " 
                  << deviceProp.warpSize << std::endl;
        std::cout << "Max threads per block: " 
                  << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads per multiprocessor: " 
                  << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Number of multiprocessors: " 
                  << deviceProp.multiProcessorCount << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
