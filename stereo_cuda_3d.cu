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


#define NUM_CHNL 3


///////////////////////////////
__device__
Float3 operator*(Float3 &a, float b)
{
    // TODO: use make_float as in (https://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4)
    Float3 res;
    res.arr[0] = a.arr[0] * b;
    res.arr[1] = a.arr[1] * b;
    res.arr[2] = a.arr[2] * b;
    return res;
}


__device__
Float3 operator-(Float3 &a, Float3 &b)
{
    Float3 res;
    res.arr[0] = a.arr[0] - b.arr[0];
    res.arr[1] = a.arr[1] - b.arr[1];
    res.arr[2] = a.arr[2] - b.arr[2];
    return res;
}


__device__
Float3 operator*(Float3 a, Float3 b)
{
    Float3 res;
    res.arr[0] = a.arr[0] * b.arr[0];
    res.arr[1] = a.arr[1] * b.arr[1];
    res.arr[2] = a.arr[2] * b.arr[2];
    return res;
}

__device__
float reduceSum(Float3 &a) {
    return a.arr[0] + a.arr[1] + a.arr[2];
}

__device__
void resetValue(Float3 &a, float val) {
    a.arr[0] = val;
    a.arr[1] = val;
    a.arr[2] = val;
}

__device__
Float3& operator+=(Float3 &first, const Float3& sec) {
    first.arr[0] += sec.arr[0];
    first.arr[1] += sec.arr[1];
    first.arr[2] += sec.arr[2];
    return first;
}
///////////////////////////////


__device__
void loadMeanStd3d(Float3 *img, int rCenter, int cCenter, Float3 &mean, Float3 &invStd, int imgHeight, int imgWidth) {
    Float3 sum;
    resetValue(sum, 0.0);
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

    mean.f = (sum * (1.0 / cnt)).f;

    Float3 varSum, diff;
    resetValue(varSum, 0.0);
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

    varSum.f = (varSum * (1.0 / cnt)).f;

    for (int channel = 0; channel < NUM_CHNL; channel++)
        invStd.arr[channel] = rsqrtf(varSum.arr[channel]);
}


__device__
float computeNCC3d(Float3 *img1, Float3 *img2, int r, int c1, int c2, int imgHeight, int imgWidth) {
    Float3 mean1, mean2, invStd1, invStd2;
    loadMeanStd3d(img1, r, c1, mean1, invStd1, imgHeight, imgWidth);
    loadMeanStd3d(img2, r, c2, mean2, invStd2, imgHeight, imgWidth);
    Float3 invStdMult = invStd1 * invStd2;

    float nccSum = 0.0;
    short itemCount = 0;
    int idx1, idx2;
    Float3 nccTerm;

    for (int dr = -HF_NCC_HEIGHT; dr <= HF_NCC_HEIGHT; dr++) {
        if (r + dr < 0 || r + dr >= imgHeight)
            continue;
        for (int dc = -HF_NCC_WIDTH; dc <= HF_NCC_WIDTH; dc++) {
            if (c1 + dc < 0 || c2 + dc < 0 || c1 + dc >= imgWidth || c2 + dc >= imgWidth)
                continue;
            idx1 = (r + dr) * imgWidth + (c1 + dc);
            idx2 = (r + dr) * imgWidth + (c2 + dc);
            nccTerm = (img1[idx1] - mean1) * (img2[idx2] - mean2) * invStdMult;
            nccSum += reduceSum(nccTerm);
            itemCount += NUM_CHNL;
        }
    }

    float ncc = nccSum / itemCount;
    return ncc;
}


__global__
void computeDisparitySet(Problem* problem) {
    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    int blockDisp = blockIdx.z;
    int threadRow = threadIdx.x;
    int threadCol = threadIdx.y;
    int threadDisp = threadIdx.z;

    int row = blockRow * blockDim.x + threadRow;
    int col = blockCol * blockDim.y + threadCol;
    int disp = blockDisp * blockDim.z + threadDisp;
    int imgHeight = problem->height;
    int imgWidth = problem->width;

    if (row >= imgHeight || col >= imgWidth || disp > MAX_DISP)
        return;

    int nccSetIdx = row * imgWidth * MAX_DISP + col * MAX_DISP + disp;

    int rightImgCol = col - disp;
    if (rightImgCol < 0 || rightImgCol >= imgWidth) {
        problem->nccSet[nccSetIdx] = -INFTY;
        return;
    }

    float ncc = computeNCC3d(problem->img1, problem->img2, row, col, rightImgCol, imgHeight, imgWidth);
    problem->nccSet[nccSetIdx] = ncc;
}


float* computeDisparityMap3D(Float3* img1, Float3* img2, int height, int width) {
    float* res;
    float* nccSet;
    cudaMalloc(&res, sizeof(float) * height * width);
    cudaMalloc(&nccSet, sizeof(float) * height * width * MAX_DISP);
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, nccSet, res);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);
    double tStart = clock();

    dim3 threadDim(DISP_THREAD_DIM, DISP_THREAD_DIM, DISP_THREAD_DIM);
    int blockCountX = (height + DISP_THREAD_DIM - 1) / DISP_THREAD_DIM;
    int blockCountY = (width + DISP_THREAD_DIM - 1) / DISP_THREAD_DIM;
    int blockCountZ = (MAX_DISP + DISP_THREAD_DIM - 1) / DISP_THREAD_DIM;
    dim3 blockDim(blockCountX, blockCountY, blockCountZ);
    computeDisparitySet<<<blockDim, threadDim>>>(problemGPU);
    cudaDeviceSynchronize();

    double tEnd = clock();
    printf("Kernel call took %.2lf ms.\n", (tEnd - tStart) / CLOCKS_PER_SEC * 1000.0);
    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(res);
    cudaFree(problemGPU);
    return resCPU;
}
