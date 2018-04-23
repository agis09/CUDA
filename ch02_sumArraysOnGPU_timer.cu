#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<cmath>
#include<time.h>
#include<string.h>
#include<time.h>

#define CHECK(call){	\
const cudaError_t error = call;		\
if (error!=cudaSuccess) {	\
	printf("Error:%s:%d, ", __FILE__, __LINE__);	\
	printf("code:%d, reason: %s\n", error,	\
			cudaGetErrorString(error));	\
	exit(1);	\
	}	\
}	\

void checkResult(float *hostRef, float *gpuRef, const int N) {
double epsilon = 1.0E-8;
bool match = 1;

for (int i = 0; i<N; i++) {
	if (abs(hostRef[i]-gpuRef[i])>epsilon) {
		match = 0;
		printf("Arrays do not match!\n");
		printf("host %5.2f gpur %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
		break;
	}
}
if (match)printf("Arrays match.\n\n");
return;
}

void initialData(float *ip, int size) {
//乱数シード生成
time_t t;
srand((unsigned)time(&t));

for (int i = 0; i<size; i++) {
	ip[i] = (float)(rand()&0xFF)/10.0f;
}
return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
	for (int idx = 0; idx<N; idx++) {
		C[idx] = A[idx]+B[idx];
	}
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	C[i] = A[i]+B[i];
}


int main(int argc, char **argv) {
	printf("%s Starting...\n", argv[0]);

	//デバイスのセットアップ
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//ベクトルのデータサイズを設定
	int nElem = 1<<24;
	printf("Vector size %d\n", nElem);

	//ホストメモリ確保
	size_t nBytes = nElem*sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);
	clock_t iStart;
	double iElaps;
	
	//ホスト側でデータを初期化
	iStart = clock();
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	iElaps = clock()-iStart;
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	//結果をチェックするためにホスト側でベクトル加算
	iStart = clock();
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	iElaps = clock()-iStart;


	//デバイスのグローバルメモリ確保
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	//ホストからデバイスへデータ転送
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

	//ホスト側でカーネルを呼び出す
	int iLen = 512;	//1024->512	
	dim3 block(iLen);
	dim3 grid((nElem+block.x-1)/block.x);

	iStart = clock();
	sumArraysOnGPU<<< grid, block>>>(d_A, d_B, d_C, nElem);
	CHECK(cudaDeviceSynchronize());
	iElaps = clock()-iStart;
	printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %fsec\n", grid.x, block.x,iElaps/CLOCKS_PER_SEC);

	//カーネルのエラーチェック
	CHECK(cudaGetLastError());

	//カーネルの結果をホスト側にコピー
	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));


	//デバイスの結果をチェック
	checkResult(hostRef, gpuRef, nElem);

	//デバイスのグローバルメモリ解放
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	//ホストのメモリ解放
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	//cudaDeviceReset();
	return 0;

}
