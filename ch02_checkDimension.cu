#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include<stdio.h>
#include<iostream>


__global__ void checkIndex(void) {
	printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim:(%d,%d,%d)\n",
		threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
		blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv) {
	//�f�[�^�v�f�̍��v��
	int nElem = 6;

	//�O���b�h�ƃu���b�N�̍\��
	dim3 block(3);
	dim3 grid((nElem+block.x-1)/block.x);

	//�O���b�h�ƃu���b�N�̃T�C�Y���z�X�g������`�F�b�N
	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

	//�O���b�h�ƃu���b�N�̃T�C�Y���f�o�C�X������`�F�b�N
	checkIndex<<<grid, block>>>();

	//�f�o�C�X�����Z�b�g
	cudaDeviceReset();

	return 0;


}