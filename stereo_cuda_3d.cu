#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector_functions.h>
#include "stereo_cuda_3d.h"


#define USE_NCC 1
#define USE_SQRT_APPROX 1

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

    float varSum, diff;
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

    varSum = varSum / cnt;
    invStd = rsqrtf(varSum);
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
    short itemCount = 0;
    float nccTerm;

    int disp = threadIdx.x;

    for (int br = 0; br < NCC_HEIGHT; br++) {
        for (int bc = 0; bc < NCC_WIDTH; bc++) {
            int bc2 = bc + (MAX_DISP - disp);
            if (buff1[br][bc] > -0.5 && buff2[br][bc2] > -0.5) {
                nccTerm = (buff1[br][bc] - mean1) * (buff2[br][bc2] - mean2) * invStdMult;
                nccSum += nccTerm;
                itemCount++;
            }
        }
    }

    float ncc = nccSum / itemCount;
    return ncc;
}


// Internally synced
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

    __syncthreads();
}


// Internally synced
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

    __syncthreads();
}


__global__
void computeDisparityNCC(Problem* problem) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int disp = threadIdx.x;;
    int imgHeight = problem->height;
    int imgWidth = problem->width;
    int nccSetIdx = row * imgWidth * MAX_DISP + col * MAX_DISP + disp;

    int rightImgCol = col - disp;
    if (rightImgCol < 0 || rightImgCol >= imgWidth) {
        problem->nccSet[nccSetIdx] = -INFTY;
        return;
    }

    __shared__ float buff1[NCC_HEIGHT][NCC_WIDTH];
    __shared__ float buff2[NCC_HEIGHT][NCC_WIDTH + MAX_DISP];

    loadIntoBuffer1(problem->img1, buff1, row, col, imgHeight, imgWidth);
    loadIntoBuffer2(problem->img2, buff2, row, col, imgHeight, imgWidth);

    float ncc = computeNCC3d(buff1, buff2, problem->meanInvStdCache, row, col, rightImgCol, imgHeight, imgWidth);
    problem->nccSet[nccSetIdx] = ncc;
}


__global__
void computeDisparity(Problem* problem) {
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


float* computeDisparityMap3D(float* img1, float* img2, int height, int width) {
    float* res;
    float* nccSet;
    float* meanInvStdCache;
    cudaMalloc(&res, sizeof(float) * height * width);
    cudaMalloc(&nccSet, sizeof(float) * height * width * MAX_DISP);
    cudaMalloc(&meanInvStdCache, sizeof(float) * height * width * 4);
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, nccSet, meanInvStdCache, res);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);
    double tStart = clock();

    // Kernel 1
    cacheMeanInvStd<<<height, width>>>(problemGPU);
    cudaDeviceSynchronize();

    // Kernel 2
    dim3 blockDim(MAX_DISP);
    dim3 gridDim(width, height);
    computeDisparityNCC<<<gridDim, blockDim>>>(problemGPU);
    cudaDeviceSynchronize();

    // Kernel 3
    computeDisparity<<<height, width>>>(problemGPU);
    cudaDeviceSynchronize();

    double tEnd = clock();
    printf("Kernel call took %.2lf ms.\n", (tEnd - tStart) / CLOCKS_PER_SEC * 1000.0);
    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(meanInvStdCache);
    cudaFree(nccSet);
    cudaFree(res);
    cudaFree(problemGPU);
    return resCPU;
}

