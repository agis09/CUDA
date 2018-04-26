#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<cmath>

int main(int argc, char **argv) {
	printf("%s Starting...\n,argv[0]");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id!=cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			(int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	if (deviceCount==0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev = 0, driverVersion = 0, runtimeVersion = 0;

	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("CUDA Driver Versin / Runtime Version	%d.%d / %d.%d",
		driverVersion/1000, (driverVersion%100)/10,
		runtimeVersion/1000, (runtimeVersion%100)/10);
	printf("CUDA Capability Mafor/Minor version number:	%d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("Total amount of global memory:	%.2f MBytes (%llu bytes)\n",
		(float)deviceProp.totalGlobalMem/(pow(1024.0, 3)),
		(unsigned long long)deviceProp.totalConstMem);
	printf("GPU Clock rate:	%.0f MHz (%0.2 GHz)\n",
		deviceProp.clockRate * 1e-3f, deviceProp.clockRate*1e-6f);
	printf("Memory Clock rate:	%.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
	printf("Memory Bus Width:	%d-bit\n", deviceProp.memoryBusWidth);
	if (deviceProp.l2CacheSize) {
		printf("L2 Cache Size:	%d bytes\n", deviceProp.l2CacheSize);
	}

	/*
				......
				
	*/

}
