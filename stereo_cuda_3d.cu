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
float computeNCC3d(float *img1, float *img2, int r, int c1, int c2, int imgHeight, int imgWidth) {
    float mean1, mean2, invStd1, invStd2;
    loadMeanStd3d(img1, r, c1, mean1, invStd1, imgHeight, imgWidth);
    loadMeanStd3d(img2, r, c2, mean2, invStd2, imgHeight, imgWidth);
    float invStdMult = invStd1 * invStd2;

    float nccSum = 0.0;
    short itemCount = 0;
    int idx1, idx2;
    float nccTerm;

    for (int dr = -HF_NCC_HEIGHT; dr <= HF_NCC_HEIGHT; dr++) {
        if (r + dr < 0 || r + dr >= imgHeight)
            continue;
        for (int dc = -HF_NCC_WIDTH; dc <= HF_NCC_WIDTH; dc++) {
            if (c1 + dc < 0 || c2 + dc < 0 || c1 + dc >= imgWidth || c2 + dc >= imgWidth)
                continue;
            idx1 = (r + dr) * imgWidth + (c1 + dc);
            idx2 = (r + dr) * imgWidth + (c2 + dc);
            nccTerm = (img1[idx1] - mean1) * (img2[idx2] - mean2) * invStdMult;
            nccSum += nccTerm;
            itemCount++;
        }
    }

    float ncc = nccSum / itemCount;
    return ncc;
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

    float ncc = computeNCC3d(problem->img1, problem->img2, row, col, rightImgCol, imgHeight, imgWidth);

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


float* computeDisparityMap3D(float* img1, float* img2, int height, int width) {
    float* res;
    float* nccSet;
    cudaMalloc(&res, sizeof(float) * height * width);
    cudaMalloc(&nccSet, sizeof(float) * height * width * MAX_DISP);
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, nccSet, res);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);
    double tStart = clock();

    // Kernel 1
    dim3 blockDim(MAX_DISP);
    dim3 gridDim(width, height);
    computeDisparityNCC<<<gridDim, blockDim>>>(problemGPU);
    cudaDeviceSynchronize();

    // Kernel 2
    computeDisparity<<<height, width>>>(problemGPU);
    cudaDeviceSynchronize();

    double tEnd = clock();
    printf("Kernel call took %.2lf ms.\n", (tEnd - tStart) / CLOCKS_PER_SEC * 1000.0);
    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(res);
    cudaFree(problemGPU);
    return resCPU;
}

