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

void checkResult(float *hostRef, float *gpuRef, const int N) {
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i=0; i<N; i++) {
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
	//�����V�[�h����
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
	int i = threadIdx.x;
	C[i] = A[i]+B[i];
}

int main(int argc, char **argv) {
	printf("%s Starting...\n", argv[0]);

	//�f�o�C�X�̃Z�b�g�A�b�v
	int dev = 0;
	cudaSetDevice(dev);

	//�x�N�g���̃f�[�^�T�C�Y��ݒ�
	int nElem = 32;
	printf("Vector size %d\n", nElem);

	//�z�X�g�������m��
	size_t nBytes = nElem*sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	//�z�X�g���Ńf�[�^��������
	initialData(h_A, nElem);
	initialData(h_B, nElem);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	//�f�o�C�X�̃O���[�o���������m��
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	//�z�X�g����f�o�C�X�փf�[�^�]��
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);

	//�z�X�g���ŃJ�[�l�����Ăяo��
	dim3 block(nElem);
	dim3 grid(1);

	sumArraysOnGPU<<< grid, block>>>(d_A, d_B, d_C, nElem);
	printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);

	//�J�[�l���̌��ʂ��z�X�g���ɃR�s�[
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	//���ʂ��`�F�b�N���邽�߂Ƀz�X�g���Ńx�N�g�������Z
	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	//�f�o�C�X�̌��ʂ��`�F�b�N
	checkResult(hostRef, gpuRef, nElem);

	//�f�o�C�X�̃O���[�o�����������
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//�z�X�g�̃��������
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaDeviceReset();
	return 0;

}
