#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<cmath>
#include<time.h>
#include<string.h>

#define CHECK(call){	\
const cudaError_t error = call;		\
if (error!=cudaSuccess) {	\
	printf("Error:%s:%d, ", __FILE__, __LINE__);	\
	printf("code:%d, reason: %s\n", error,	\
			cudaGetErrorString(error));	\
	exit(1);	\
	}	\
}	\

void printMatrix(int *C, const int nx, const int ny) {
	int *ic = C;
	printf("\nMatrix: (%d.%d)\n", nx, ny);
	for (int iy = 0; iy<ny; iy++) {
		for (int ix = 0; ix<nx; ix++) {
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
	return;
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC,int nx,int ny) {
	int ix = threadIdx.x+blockIdx.x*blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	unsigned int idx = iy*nx+ix;

	if (ix<nx&&iy<ny) {
		MatC[idx] = MatA[idx]+MatB[idx];
	}
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

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
	float *ia = A;
	float *ib = B;
	float *ic = C;

	for(int iy = 0; iy<ny; iy++) {
		for (int ix = 0; ix<nx; ix++) {
			ic[ix] = ia[ix]+ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
	return;
}

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
}

int main(int argc, char **argv) {
	printf("%s Starting...\n", argv[0]);

	//デバイスのセットアップ
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//行列のデータサイズを設定
	int nx = 1<<14;
	int ny = 1<<14;

	int nxy = nx*ny;
	int nBytes = nxy*sizeof(float);
	printf("Matrix size: nx%d ny%d\n", nx, ny);

	//ホストメモリ確保
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	clock_t iStart;

	

	//ホスト側でデータを初期化
	iStart = clock();
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	double iElaps = clock()-iStart;


	
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);



	//結果をチェックするためにホスト側で行列を加算
	iStart = clock();
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	iElaps = clock()-iStart;


	//デバイスのグローバルメモリを確保
	float *d_MatA, *d_MatB, *d_MatC;
	CHECK(cudaMalloc((void **)&d_MatA, nBytes));
	CHECK(cudaMalloc((void **)&d_MatB, nBytes));
	CHECK(cudaMalloc((void **)&d_MatC, nBytes));

	//ホストからデバイスデータを転送
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));


	//ホスト側でカーネルを呼び出す
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

	iStart = clock();
	sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
	CHECK(cudaDeviceSynchronize());
	iElaps = clock()-iStart;
	printf("sumMatrixOnGPU2D<<<(%d,%d),(%d,%d)>>>elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps/CLOCKS_PER_SEC);

	//カーネルエラーチェック
	CHECK(cudaGetLastError());

	//カーネルの結果をホスト側にコピー
	CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

	//デバイスの結果をチェック
	checkResult(hostRef, gpuRef, nxy);

	//デバイスのグローバルメモリを解放
	CHECK(cudaFree(d_MatA));
	CHECK(cudaFree(d_MatB));
	CHECK(cudaFree(d_MatC));

	//ホストのメモリを解放
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	//デバイスリセット
	CHECK(cudaDeviceReset());

	return 0;
}