#include<cuda_runtime.h>
#include<stdio.h>

int main(int argc, char **arg) {
	//データ要素の合計数を定義
	int nElem = 1024;
	//グリッドとブロックの構造を定義
	dim3 block(1024);
	dim3 grid((nElem+block.x-1)/block.x);
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//ブロックをリセット
	block.x = 512;
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);


	//ブロックをリセット
	block.x = 256;
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//ブロックをリセット
	block.x = 128;
	grid.x = (nElem+block.x-1)/block.x;
	printf("grid.x %d block.x %d \n", grid.x, block.x);

	//デバイスをリセット
	cudaDeviceReset();
	return 0;
}