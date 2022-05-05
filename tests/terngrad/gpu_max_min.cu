#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <cooperative_groups.h>

#if __DEVICE_EMULATION__
#define DEBUG_SYNC sync(cooperative_groups::this_thread_block());
#else
#define DEBUG_SYNC
#endif

#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)
#else
#define int_mult(x,y)	x*y
#endif

#define inf 0x7f800000


const int blockSize1 = 2048;
/*const int blockSize2 = 8192;
const int blockSize3 = 16384;
const int blockSize4 = 32768;
const int blockSize5 = 65536;*/

const int threads = 64;

unsigned long long int get_clock() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (unsigned long long int)tv.tv_usec + 1000000*tv.tv_sec;
}

__device__ void warp_reduce_max(volatile float smem[64]) {
	smem[threadIdx.x] = smem[threadIdx.x+32] > smem[threadIdx.x] ?
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+16] > smem[threadIdx.x] ?
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+8] > smem[threadIdx.x] ?
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+4] > smem[threadIdx.x] ?
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+2] > smem[threadIdx.x] ?
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+1] > smem[threadIdx.x] ?
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;
}

__device__ void warp_reduce_min(volatile float smem[64]) {
	smem[threadIdx.x] = smem[threadIdx.x+32] < smem[threadIdx.x] ?
						smem[threadIdx.x+32] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+16] < smem[threadIdx.x] ?
						smem[threadIdx.x+16] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+8] < smem[threadIdx.x] ?
						smem[threadIdx.x+8] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+4] < smem[threadIdx.x] ?
						smem[threadIdx.x+4] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+2] < smem[threadIdx.x] ?
						smem[threadIdx.x+2] : smem[threadIdx.x]; DEBUG_SYNC;
	smem[threadIdx.x] = smem[threadIdx.x+1] < smem[threadIdx.x] ?
						smem[threadIdx.x+1] : smem[threadIdx.x]; DEBUG_SYNC;
}

template<int threads>
__global__ void find_min_max_dynamic(float* in, float* out, int n, int start_adr, int num_blocks) {

	volatile __shared__ float smem_min[64];
	volatile __shared__ float smem_max[64];

	int tid = threadIdx.x + start_adr;

	float max = -inf;
	float min = inf;
	float val;

	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;

	// tail part
	int mult = 0;
	for(int i = 1; mult + tid < n; i++) {
		val = in[tid + mult];
		min = val < min ? val : min;
		max = val > max ? val : max;
		mult = int_mult(i, threads);
	}

	// previously reduced MIN part
	mult = 0;
	int i;
	for(i = 1; mult + threadIdx.x < num_blocks; i++) {
		val = out[threadIdx.x + mult];
		min = val < min ? val : min;
		mult = int_mult(i, threads);
	}

	// MAX part
	for(; mult + threadIdx.x < num_blocks*2; i++) {
		val = out[threadIdx.x + mult];
		max = val > max ? val : max;
		mult = int_mult(i, threads);
	}


	if(threads == 32) {
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;
	}

	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;

	sync(cooperative_groups::this_thread_block());

	if(threadIdx.x < 32) {
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if(threadIdx.x == 0) {
		out[blockIdx.x] = smem_min[threadIdx.x];
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
	}
}

template<int els_per_block, int threads>
__global__ void find_min_max(float* in, float* out) {
	volatile __shared__ float smem_min[64];
	volatile __shared__ float smem_max[64];

	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	float max = -inf;
	float min = inf;
	float val;

	const int iters = els_per_block/threads;

#pragma unroll
	for (int i = 0; i < iters; i++) {
		val = in[tid + i*threads];
		min = val < min ? val : min;
		max = val > max ? val : max;
	}

	if (threads == 32) {
		smem_min[threadIdx.x+32] = 0.0f;
		smem_max[threadIdx.x+32] = 0.0f;
	}

	smem_min[threadIdx.x] = min;
	smem_max[threadIdx.x] = max;

	sync(cooperative_groups::this_thread_block());

	if (threadIdx.x < 32) {
		warp_reduce_min(smem_min);
		warp_reduce_max(smem_max);
	}
	if (threadIdx.x == 0) {
		out[blockIdx.x] = smem_min[threadIdx.x]; // out[0] == ans
		out[blockIdx.x + gridDim.x] = smem_max[threadIdx.x];
	}
}

void findBlockSize(int* whichSize, int* num_el) {
	const float pretty_big_number = 24.0f*1024.0f*1024.0f;
	float ratio = float((*num_el))/pretty_big_number;

	if(ratio > 0.8f)
		(*whichSize) =  5;
	else if(ratio > 0.6f)
		(*whichSize) =  4;
	else if(ratio > 0.4f)
		(*whichSize) =  3;
	else if(ratio > 0.2f)
		(*whichSize) =  2;
	else
		(*whichSize) =  1;
}

void MIN_MAX(float* d_in, int num_els, float *min, float *max) {
	int whichSize = -1;
	findBlockSize(&whichSize, &num_els);

	int block_size = powf(2, whichSize - 1) * blockSize1;
	int num_blocks = num_els / block_size;
	int tail = num_els - num_blocks * block_size;
	int start_adr = num_els - tail;

	float *d_out;
	int allocated_size = num_blocks == 0 ? 2 : num_blocks * 2;
	cudaMalloc((void**)&d_out, allocated_size * sizeof(float));

	if(whichSize == 1)
		find_min_max<blockSize1,threads><<< num_blocks, threads>>>(d_in, d_out);
	else if(whichSize == 2)
		find_min_max<blockSize1*2,threads><<< num_blocks, threads>>>(d_in, d_out);
	else if(whichSize == 3)
		find_min_max<blockSize1*4,threads><<< num_blocks, threads>>>(d_in, d_out);
	else if(whichSize == 4)
		find_min_max<blockSize1*8,threads><<< num_blocks, threads>>>(d_in, d_out);
	else
		find_min_max<blockSize1*16,threads><<< num_blocks, threads>>>(d_in, d_out);

	find_min_max_dynamic<threads><<< 1, threads>>>(d_in, d_out, num_els, start_adr, num_blocks);
	cudaMemcpy(min, d_out, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(max, d_out+1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_out);
}

float cpu_min(float* in, int num_els) {
	float min = inf;
	for(int i = 0; i < num_els; i++)
		min = in[i] < min ? in[i] : min;
	return min;
}
float cpu_max(float* in, int num_els) {
	float max = -inf;
	for(int i = 0; i < num_els; i++)
		max = in[i] > max ? in[i] : max;
	return max;
}

void cpu_min_max(float* in, int num_els) {
	float max = -inf;
	float min = inf;
	for (int i = 0; i < num_els; i++) {
		min = in[i] < min ? in[i] : min;
		max = in[i] > max ? in[i] : max;
	}
}

unsigned long long int my_min_max_test(int num_els) {

	unsigned long long int start;
	unsigned long long int delta;

	int testIterations = 100;

	int size = num_els*sizeof(float);

	float* d_in;

	float* in = (float*)malloc(size);

	float min = 0.0, max = 0.0;

	for(int i = 0; i < num_els; i++) {
		in[i] = rand()&1;
	}

	in[0] = 34.0f;
	in[1] = 55.0f;
	in[2] = -42.0f;

	cudaMalloc((void**)&d_in, size);

	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

	start = get_clock();
	for(int i = 0; i < testIterations; i++) {
		min = max = 0.0;
		MIN_MAX(d_in, num_els, &min, &max);
	}
	cudaDeviceSynchronize();
	delta = get_clock() - start;

	float dt = float(delta)/float(testIterations);

	float throughput = num_els*sizeof(float)*0.001f/(dt);
	printf("%16.0d \t\t %0.2f \t\t %0.2f % \t\t %0.1f \t\t %s \t\t %0.3f \t\t %0.3f ", num_els, throughput,
		(throughput/1408.0f)*100.0f,dt,  (cpu_min(in,num_els) == min && cpu_max(in,num_els) == max) ? "Pass" : "Fail", cpu_min(in,num_els), max);

	start = get_clock();
	for (int i = 0; i < 20; i++) {
		cpu_min_max(in, num_els);
	}
	delta = get_clock() - start;

	dt = float(delta)/float(20);
	printf("\t %0.1f\n", dt);

	cudaFree(d_in);
	free(in);

	return delta;
}

int main(int argc, char* argv[]) {
	printf("%16s \t\t [GB/s] \t\t [perc] \t\t [usec] \t [test] \t [min] \t\t [max] \t\t [cpu_usec] \n","N");

	for(int i = 32; i <= 128*1024*1024; i = i*2) {
		my_min_max_test(i);
	}

	printf("\nNon-base 2 tests! \n");
	printf("N \t\t [GB/s] \t\t [perc] \t\t [usec] \t [test] \t [min] \t\t [max] \t\t [cpu_usec] \n");

	my_min_max_test(14*1024*1024+38);
	my_min_max_test(14*1024*1024+55);
	my_min_max_test(18*1024*1024+1232);
	my_min_max_test(7*1024*1024+94854);

	for(int i = 0; i < 4; i++) {
		float ratio = float(rand())/float(RAND_MAX);
		ratio = ratio >= 0 ? ratio : -ratio;
		int big_num = ratio*18*1e6;
		my_min_max_test(big_num);
	}
	return 0;
}
