#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <process.h>


#define OFFSET(row,col,xDim) ((row) * (xDim) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


__global__ void mySgemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, 
    const int M, const int N, const int P)
{
	const int bm = 128;
    const int bn = 128;
    const int bp = 8;
    const int tm = 8;
    const int tn = 8;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ float A_shared[2][bp][bm];// 矩阵A在共享内存中按列存储
    __shared__ float B_shared[2][bp][bn]; 
	float register_load_a[4], register_load_b[4];//
	float register_compute_a[tm], register_compute_b[tn];
	float result_compute[tm][tn] = { 0.0 };
	
	//本block需要搬运的矩阵A的行起始位置和矩阵B的列起始位置
	int row_block_start = blockIdx.y * bm;
	int col_block_start = blockIdx.x * bn;

    // 本线程需要搬运的数据在共享内存中的行列
    int row_a_smem = tid / 2;
    int col_a_smem = (tid % 2) * 4;
    int row_b_smem = tid / 32;
    int col_b_smem = (tid % 32) * 4;

    // 本线程需要搬运的矩阵A和矩阵B的数据行列起始
    int row_load_a = row_block_start + row_a_smem;
    int col_load_a = col_a_smem;
    int row_load_b = row_b_smem;
    int col_load_b = col_block_start + col_b_smem;

	//每个线程搬运四个数据，先搬运到寄存器中，再由寄存器搬运到共享内存中，
	//每次block只搬运矩阵A的bp列和矩阵B的bp行数据
	//再由寄存器搬运到共享内存中
	FLOAT4(register_load_a[0]) = FLOAT4(A[OFFSET(row_load_a, col_load_a, P)]);
	FLOAT4(register_load_b[0]) = FLOAT4(B[OFFSET(row_load_b, col_load_b, N)]);
	A_shared[0][col_a_smem][row_a_smem] = register_load_a[0];
	A_shared[0][col_a_smem + 1][row_a_smem] = register_load_a[1];
	A_shared[0][col_a_smem + 2][row_a_smem] = register_load_a[2];
	A_shared[0][col_a_smem + 3][row_a_smem] = register_load_a[3];
	FLOAT4(B_shared[0][row_b_smem][col_b_smem]) = FLOAT4(register_load_b[0]);

	//大循环
	for (int pp = 1; pp < (P+bp-1) / bp; pp++)
	{
		int d = (pp - 1) % 2; //double buffering的指示变量
		int d_next = pp % 2; //double buffering的指示变量

		//从全局内存中取数据,放入寄存器中
        col_load_a = pp * bp + col_a_smem;
        row_load_b = pp * bp + row_b_smem;
        FLOAT4(register_load_a[0]) = FLOAT4(A[OFFSET(row_load_a, col_load_a, P)]);
        FLOAT4(register_load_b[0]) = FLOAT4(B[OFFSET(row_load_b, col_load_b, N)]);

	    //从共享内存中取数据并计算
		for (int ppp = 0; ppp < bp; ppp++)
		{
			FLOAT4(register_compute_a[0]) = FLOAT4(A_shared[d][ppp][threadIdx.y * tm/2]);
			FLOAT4(register_compute_a[4]) = FLOAT4(A_shared[d][ppp][threadIdx.y * tm/2 + bm/2]);
			FLOAT4(register_compute_b[0]) = FLOAT4(B_shared[d][ppp][threadIdx.x * tn/2]);
			FLOAT4(register_compute_b[4]) = FLOAT4(B_shared[d][ppp][threadIdx.x * tn/2 + bn/2]);

			for (int i = 0; i < tm; i++)
			{
				for (int j = 0; j < tn; j++)
				{
					result_compute[i][j] += register_compute_a[i] * register_compute_b[j];
				}
			}
		}

		//将寄存器中的数据放入共享内存中
        A_shared[d_next][col_a_smem][row_a_smem] = register_load_a[0];
        A_shared[d_next][col_a_smem + 1][row_a_smem] = register_load_a[1];
        A_shared[d_next][col_a_smem + 2][row_a_smem] = register_load_a[2];
        A_shared[d_next][col_a_smem + 3][row_a_smem] = register_load_a[3];
        FLOAT4(B_shared[d_next][row_b_smem][col_b_smem]) = FLOAT4(register_load_b[0]);

		__syncthreads();
	}

	//最后一个计算
	for (int ppp = 0; ppp < bp; ppp++)
	{
		FLOAT4(register_compute_a[0]) = FLOAT4(A_shared[1][ppp][threadIdx.y * tm / 2]);
		FLOAT4(register_compute_a[4]) = FLOAT4(A_shared[1][ppp][threadIdx.y * tm / 2 + bm / 2]);
		FLOAT4(register_compute_b[0]) = FLOAT4(B_shared[1][ppp][threadIdx.x * tn / 2]);
		FLOAT4(register_compute_b[4]) = FLOAT4(B_shared[1][ppp][threadIdx.x * tn / 2 + bn / 2]);

		for (int i = 0; i < tm; i++)
		{
			for (int j = 0; j < tn; j++)
			{
				result_compute[i][j] += register_compute_a[i] * register_compute_b[j];
			}
		}
	}


	//将数据写回大的矩阵C
	for (int i = 0; i < tm/2; i++)
	{
		int rowC = row_block_start + threadIdx.y * tm / 2 + i;
		int colC = col_block_start + threadIdx.x * tn / 2;
		int addrC = OFFSET(rowC, colC, N);
		FLOAT4(C[addrC]) = FLOAT4(result_compute[i][0]);
		FLOAT4(C[addrC + bn/2]) = FLOAT4(result_compute[i][4]);
	}
	for (int i = 0; i < tm / 2; i++)
	{
		int rowC = row_block_start + bm/2 + threadIdx.y * tm / 2 + i;
		int colC = col_block_start + threadIdx.x * tn / 2;
		int addrC = OFFSET(rowC, colC, N);
		FLOAT4(C[addrC]) = FLOAT4(result_compute[tm/2 + i][0]);
		FLOAT4(C[addrC + bn / 2]) = FLOAT4(result_compute[tm/2 + i][4]);
	}
}

void cpuSgemm(
    float* a, float* b, float* c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


float testMaxError(
    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* h_a, * h_b, * h_c, * d_a, * d_b, * d_c, * h_d_c;
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float*)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm << <gridDim, blockDim >> > (d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testCublasMaxError(const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* h_a, * h_b, * h_c, * d_a, * d_b, * d_c, * h_d_c;
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float*)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm << <gridDim, blockDim >> > (d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

int main() {

    const int M_list[15] = { 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384 };
    const int N_list[15] = { 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384 };
    // const int K_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = { 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024 };
    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nKernal = cublas\n");

        {
            const int M = 512, N = 512, K = 512;
            float max_error = testCublasMaxError(M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 15;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testCublasPerformance(M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    

    {
        printf("\nKernal = mySgemm\n");

        const int BM = 128, BN = 128, TM = 8, TN = 8;
        void (*gpuSgemm) (float*, float*, float*, const int, const int, const int) =
            mySgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN / TN, BM / TM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = testMaxError(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 15;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN / TN, BM / TM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

    return 0;
}
