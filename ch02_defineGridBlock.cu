#include<cuda_runtime.h>
#include<stdio.h>

int main(int argc, char **arg) {
	//�f�[�^�v�f�̍��v�����`
	int nElem = 1024;
	//�O���b�h�ƃu���b�N�̍\�����`
	dim3 block(1024);
	dim3 grid((nElem+block.x-1)/block.x);
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//�u���b�N�����Z�b�g
	block.x = 512;
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);


	//�u���b�N�����Z�b�g
	block.x = 256;
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//�u���b�N�����Z�b�g
	block.x = 128;
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//�f�o�C�X�����Z�b�g
	cudaDeviceReset();
	return 0;
}