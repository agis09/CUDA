#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void helloFromGPU() {
	if (threadIdx.x==5) {
		printf("Hello World from GPU thread %d\n",threadIdx.x);
	}
}

int main(int argc, char **argv) {
	printf("Hello from cpu\n");
	helloFromGPU<<<1, 10>>>();
	cudaDeviceReset();
	return 0;
}