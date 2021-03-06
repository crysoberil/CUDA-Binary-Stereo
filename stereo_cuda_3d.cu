#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector_functions.h>
#include "stereo_cuda_3d.h"


#define DISPLAY_KERNEL_CALL_TIME 0

#define DISP_THREAD_DIM 8

//#define BLOCK_SIZE 32
#define NCC_HEIGHT 3
#define NCC_WIDTH 7
#define HF_NCC_HEIGHT (NCC_HEIGHT / 2)
#define HF_NCC_WIDTH (NCC_WIDTH / 2)
//#define WIDE_PATCH_H (BLOCK_SIZE + NCC_HEIGHT - 1)
//#define WIDE_PATCH_W (BLOCK_SIZE + NCC_WIDTH - 1)
//#define WIDE_PATCH_ELM (WIDE_PATCH_H * WIDE_PATCH_W)
#define MAX_DISP 128
#define INFTY (1 << 29)


__device__
void loadMeanStd3d(float *img, int rCenter, int cCenter, float &mean, float &invStd, int imgHeight, int imgWidth) {
    float sum = 0;
    int cnt = 0;

    for (int r = rCenter - HF_NCC_HEIGHT; r <= rCenter + HF_NCC_HEIGHT; r++) {
        if (r < 0 || r >= imgHeight)
            continue;
        for (int c = cCenter - HF_NCC_WIDTH; c <= cCenter + HF_NCC_WIDTH; c++) {
            if (c < 0 || c >= imgWidth)
                continue;
            sum += img[r * imgWidth + c];
            cnt++;
        }
    }

    mean = sum / cnt;

    double varSum = 0.0;
    float diff;
    for (int r = rCenter - HF_NCC_HEIGHT; r <= rCenter + HF_NCC_HEIGHT; r++) {
        if (r < 0 || r >= imgHeight)
            continue;
        for (int c = cCenter - HF_NCC_WIDTH; c <= cCenter + HF_NCC_WIDTH; c++) {
            if (c < 0 || c >= imgWidth)
                continue;
            diff = img[r * imgWidth + c] - mean;
            varSum += (diff * diff);
        }
    }

    if (varSum < 1e-5 && varSum > -1e-5)
        invStd = 1e-5;
    else {
        varSum = varSum / cnt;
        invStd = rsqrt(varSum);
    }
}


__device__
float computeNCC3d(float buff1[][NCC_WIDTH], float buff2[NCC_HEIGHT][NCC_WIDTH + MAX_DISP], float* meanInvStdCache, int r, int c1, int c2, int imgHeight, int imgWidth) {
    float* cache1 = meanInvStdCache + (r * imgWidth + c1) * 4;
    float* cache2 = meanInvStdCache + (r * imgWidth + c2) * 4 + 2;

    float mean1 = *cache1;
    float mean2 = *cache2;
    float invStd1 = cache1[1];
    float invStd2 = cache2[1];

    float invStdMult = invStd1 * invStd2;
    float nccSum = 0.0;
    int itemCount = 0;
    float nccTerm;

    int disp = threadIdx.x;

    for (int br = 0; br < NCC_HEIGHT; br++) {
        for (int bc = 0; bc < NCC_WIDTH; bc++) {
            int bc2 = bc + (MAX_DISP - disp);
            if (buff1[br][bc] > -0.5 && buff2[br][bc2] > -0.5) {
                nccTerm = (buff1[br][bc] - mean1) * (buff2[br][bc2] - mean2) * invStdMult;
                if (nccTerm > -1e2) {
                    nccSum += nccTerm;
                    itemCount++;
                }
            }
        }
    }

    float ncc = nccSum / itemCount;
    return ncc;
}


__device__
void loadIntoBuffer1(float* img, float buff[][NCC_WIDTH], int row, int col, int imgHeight, int imgWidth) {
    if (threadIdx.x == 0) {
        int r, c;
        for (int dr = -HF_NCC_HEIGHT; dr <= HF_NCC_HEIGHT; dr++) {
            for (int dc = -HF_NCC_WIDTH; dc <= HF_NCC_WIDTH; dc++) {
                r = row + dr;
                c = col + dc;
                if (r >= 0 && r < imgHeight && c >= 0 && c < imgWidth)
                    buff[dr + HF_NCC_HEIGHT][dc + HF_NCC_WIDTH] = img[r * imgWidth + c];
                else
                    buff[dr + HF_NCC_HEIGHT][dc + HF_NCC_WIDTH] = -1.0;
            }
        }
    }
}


__device__
void loadIntoBuffer2(float* img, float buff[NCC_HEIGHT][NCC_WIDTH + MAX_DISP], int row, int col, int imgHeight, int imgWidth) {
    int cMin = col - MAX_DISP - HF_NCC_WIDTH;
    int cMax = col + HF_NCC_WIDTH;

    int disp = threadIdx.x;

    for (int r = row - HF_NCC_HEIGHT; r <= row + HF_NCC_HEIGHT; r++) {
        for (int c = cMin + disp; c <= cMax; c += MAX_DISP) {
            if (r >= 0 && r < imgHeight && c >= 0 && c < imgWidth)
                buff[r - (row - HF_NCC_HEIGHT)][c - cMin] = img[r * imgWidth + c];
            else
                buff[r - (row - HF_NCC_HEIGHT)][c - cMin] = -1.0;
        }
    }
}


__global__
void computeDisparityNCC(Problem* problem) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int disp = threadIdx.x;
    int imgHeight = problem->height;
    int imgWidth = problem->width;
    int nccSetIdx = row * imgWidth * MAX_DISP + col * MAX_DISP + disp;

    int rightImgCol = col - disp;
    bool cellIsValid = rightImgCol >= 0 && rightImgCol < imgWidth;
    if (!cellIsValid)
        problem->nccSet[nccSetIdx] = -INFTY;

    __shared__ float buff1[NCC_HEIGHT][NCC_WIDTH];
    __shared__ float buff2[NCC_HEIGHT][NCC_WIDTH + MAX_DISP];

    loadIntoBuffer1(problem->img1, buff1, row, col, imgHeight, imgWidth); // Regardless of whether cell is valid or not.
    loadIntoBuffer2(problem->img2, buff2, row, col, imgHeight, imgWidth); // Regardless of whether cell is valid or not.

    __syncthreads();

    if (cellIsValid) {
        float ncc = computeNCC3d(buff1, buff2, problem->meanInvStdCache, row, col, rightImgCol, imgHeight, imgWidth);
        problem->nccSet[nccSetIdx] = ncc;
    }
}


__global__
void computeDisparity(Problem* problem) {
    // Sequential reduction per pixel.
    int row = blockIdx.x;
    int col = threadIdx.x;
    int imgWidth = problem->width;

    int nccSetOffset = row * imgWidth * MAX_DISP + col * MAX_DISP;
    float* nccSet = problem->nccSet + nccSetOffset;

    float bestNCC = -1e10;
    int bestDisp = 0;

    for (int i = 0; i < MAX_DISP; i++) {
        if (nccSet[i] > bestNCC) {
            bestNCC = nccSet[i];
            bestDisp = i;
        }
    }

    problem->res[row * imgWidth + col] = bestDisp;
}


__global__
void cacheMeanInvStd(Problem* problem) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int imgWidth = problem->width;
    float *cache = problem->meanInvStdCache + (row * imgWidth + col) * 4;
    float mean1, invstd1, mean2, invstd2;
    loadMeanStd3d(problem->img1, row, col, mean1, invstd1, problem->height, imgWidth);
    loadMeanStd3d(problem->img2, row, col, mean2, invstd2, problem->height, imgWidth);
    *cache = mean1;
    cache[1] = invstd1;
    cache[2] = mean2;
    cache[3] = invstd2;
}


__global__
void computeDisparityParallelReduction(Problem* problem) {
    // Parallel reduction per pixel.
    int row = blockIdx.y;
    int col = blockIdx.x;
    int threadNo = threadIdx.x;
    int imgWidth = problem->width;
    int idxOffset = (row * imgWidth + col) * MAX_DISP;
    float* nccSet = problem->nccSet;
    int numThreads = blockDim.x;

    __shared__ float bestNCC[MAX_DISP];
    __shared__ int bestWho[MAX_DISP];

    bestNCC[threadNo] = nccSet[idxOffset + threadNo];
    bestWho[threadNo] = threadNo;
    bestNCC[threadNo + numThreads] = nccSet[idxOffset + numThreads + threadNo];
    bestWho[threadNo + numThreads] = threadNo + numThreads;
    __syncthreads();

    for (int stride = numThreads; stride > 0; stride >>= 1) {
        if (threadNo < stride && bestNCC[threadNo] < bestNCC[threadNo + stride]) {
            bestNCC[threadNo] = bestNCC[threadNo + stride];
            bestWho[threadNo] = bestWho[threadNo + stride];
        }
        __syncthreads();
    }

    if (!threadNo)
        problem->res[row * imgWidth + col] = bestWho[0];
}


float* computeDisparityMap3D(float* img1, float* img2, int height, int width, float* meanInvStdCache, float* nccSet, float* res) {
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, nccSet, meanInvStdCache, res);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);
    #if (DISPLAY_KERNEL_CALL_TIME)
        double tStart = clock();
    #endif

    // Kernel 1
    cacheMeanInvStd<<<height, width>>>(problemGPU);

    // Kernel 2
    dim3 blockDim(MAX_DISP);
    dim3 gridDim(width, height);
    computeDisparityNCC<<<gridDim, blockDim>>>(problemGPU);

    // Kernel 3
    computeDisparityParallelReduction<<<gridDim, (MAX_DISP + 1) / 2>>>(problemGPU);
//    computeDisparity<<<height, width>>>(problemGPU);
    cudaDeviceSynchronize();

    #if (DISPLAY_KERNEL_CALL_TIME)
        double tEnd = clock();
        printf("Kernel call took %.2lf ms.\n", (tEnd - tStart) / CLOCKS_PER_SEC * 1000.0);
    #endif

    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(problemGPU);
    return resCPU;
}

