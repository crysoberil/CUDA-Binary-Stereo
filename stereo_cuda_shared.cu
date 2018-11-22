#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector_functions.h>
#include "stereo_cuda_shared.h"


#define USE_NCC 1
#define USE_SQRT_APPROX 1

#define BLOCK_SIZE 32
#define NCC_HEIGHT 3
#define NCC_WIDTH 7
#define HF_NCC_HEIGHT (NCC_HEIGHT / 2)
#define HF_NCC_WIDTH (NCC_WIDTH / 2)
#define WIDE_PATCH_H (BLOCK_SIZE + NCC_HEIGHT - 1)
#define WIDE_PATCH_W (BLOCK_SIZE + NCC_WIDTH - 1)
#define WIDE_PATCH_ELM (WIDE_PATCH_H * WIDE_PATCH_W)
#define MAX_DISP (BLOCK_SIZE * 4)
#define INFTY (1 << 29)

#define NUM_CHNL 3

// ###################################################

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

// ###################################################

/* Internally synced */
__device__
void loadIntoBuffer(Float3* arr, Float3 buffer[][WIDE_PATCH_W], int offsetRow, int offsetCol, int imgHeight, int imgWidth) {
    int respAlongRow = (WIDE_PATCH_H + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int respAlongCol = (WIDE_PATCH_W + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int buffIdxRow, buffIdxCol;
    int arrIdxRow, arrIdxCol;
    bool flag1, flag2;
    for (int i = 0; i < respAlongRow; i++) {
        buffIdxRow = i * BLOCK_SIZE + threadIdx.y;
        if (buffIdxRow >= WIDE_PATCH_H)
            break;
        arrIdxRow = buffIdxRow + offsetRow;
        flag1 = arrIdxRow < 0 || arrIdxRow >= imgHeight;
        for (int j = 0; j < respAlongCol; j++) {
            buffIdxCol = j * BLOCK_SIZE + threadIdx.x;
            if (buffIdxCol >= WIDE_PATCH_W)
                break;
            arrIdxCol = buffIdxCol + offsetCol;
            flag2 = arrIdxCol < 0 || arrIdxCol >= imgWidth;
            if (flag1 || flag2)
                resetValue(buffer[buffIdxRow][buffIdxCol], -1);
            else
                buffer[buffIdxRow][buffIdxCol] = arr[arrIdxRow * imgWidth + arrIdxCol];
        }
    }

    __syncthreads();
}


__device__
void flushBuffer(Float3 buffer[][WIDE_PATCH_W]) {
    for (int i = 0; i < WIDE_PATCH_H; i++) {
        for (int j = 0; j < WIDE_PATCH_W; j++)
            printf("(%.2f, %.2f, %.2f)\t", buffer[i][j].arr[0], buffer[i][j].arr[1], buffer[i][j].arr[2]);
        printf("\n");
    }
}


__device__
float inverseSqrt(float n, int iter) {
    if (n < 1e-5)
        return 1e-5;
    #if (USE_PARALLEL_DIRECTIVES)
        float x = 0.5;
        for (int i = 0; i < iter; i++)
            x -= (x * x - 1.0 / n) / (2.0 * x);
        return x;
    #else
        return sqrt(1.0 / n);
    #endif
}

__device__
void loadMeanStd(Float3 buf[][WIDE_PATCH_W], int rCenter, int cCenter, Float3 &mean, Float3 &invStd) {
    Float3 sum;
    resetValue(sum, 0.0);
    int cnt = 0;
    for (int r = rCenter - HF_NCC_HEIGHT; r <= rCenter + HF_NCC_HEIGHT; r++) {
        for (int c = cCenter - HF_NCC_WIDTH; c <= cCenter + HF_NCC_WIDTH; c++) {
            if (buf[r][c].arr[0] >= 0) {
                sum += buf[r][c];
                cnt++;
            }
        }
    }

    mean.f = (sum * (1.0 / cnt)).f;

    Float3 varSum, diff;
    resetValue(varSum, 0.0);
    for (int r = rCenter - HF_NCC_HEIGHT; r <= rCenter + HF_NCC_HEIGHT; r++) {
        for (int c = cCenter - HF_NCC_WIDTH; c <= cCenter + HF_NCC_WIDTH; c++) {
            if (buf[r][c].arr[0] >= 0) {
                diff = buf[r][c] - mean;
                varSum += (diff * diff);
            }
        }
    }

    varSum.f = (varSum * (1.0 / cnt)).f;

    for (int channel = 0; channel < NUM_CHNL; channel++)
        invStd.arr[channel] = inverseSqrt(varSum.arr[channel], 4);
}


__device__
float computeNCCShared(Float3 buf1[][WIDE_PATCH_W], Float3 buf2[][WIDE_PATCH_W], int r, int c1, int c2) {
    Float3 mean1, mean2, invStd1, invStd2;
    loadMeanStd(buf1, r, c1, mean1, invStd1);
    loadMeanStd(buf2, r, c2, mean2, invStd2);
    Float3 invStdMult = invStd1 * invStd2;

    float nccSum = 0.0;
    short itemCount = 0;
    Float3 nccTerm;

    for (int dr = -HF_NCC_HEIGHT; dr <= HF_NCC_HEIGHT; dr++) {
        for (int dc = -HF_NCC_WIDTH; dc <= HF_NCC_WIDTH; dc++) {
            if (buf1[r + dr][c1 + dc].arr[0] >= 0 && buf2[r + dr][c2 + dc].arr[0] >= 0) {
                nccTerm = (buf1[r + dr][c1 + dc] - mean1) * (buf2[r + dr][c2 + dc] - mean2) * invStdMult;
                nccSum += reduceSum(nccTerm);
                itemCount += NUM_CHNL;
            }
        }
    }

    float ncc = nccSum / itemCount;
    return ncc;
}


__device__
float computeSSDShared(Float3 buf1[][WIDE_PATCH_W], Float3 buf2[][WIDE_PATCH_W], int r, int c1, int c2) {
    float ssdSum = 0.0;
    short cnt = 0;
    Float3 diff;
    for (int dr = -HF_NCC_HEIGHT; dr <= HF_NCC_HEIGHT; dr++) {
        for (int dc = -HF_NCC_WIDTH; dc <= HF_NCC_WIDTH; dc++) {
            if (buf1[r + dr][c1 + dc].arr[0] >= 0 && buf2[r + dr][c2 + dc].arr[0] >= 0) {
                diff = buf1[r + dr][c1 + dc] - buf2[r + dr][c2 + dc];
                diff = diff * diff;
                ssdSum += reduceSum(diff);
                cnt++;
            }
        }
    }
    return -ssdSum / (cnt * NUM_CHNL);
}


__global__
void disparityKernel(Problem* problem) {
    int br = blockIdx.y;
    int bc = blockIdx.x;
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int imgHeight = problem->height;
    int imgWidth = problem->width;
    int blockLeaderRow = br * blockDim.y;
    int blockLeaderCol = bc * blockDim.x;
    int row = blockLeaderRow + tr;
    int col = blockLeaderCol + tc;
    bool cellInvalid = row >= imgHeight || col >= imgWidth;

    __shared__ Float3 buffer1[WIDE_PATCH_H][WIDE_PATCH_W];
    __shared__ Float3 buffer2[WIDE_PATCH_H][WIDE_PATCH_W];
    // Load into shared memory
    if (cellInvalid)
        __syncthreads();
    else
        loadIntoBuffer(problem->img1, buffer1, blockLeaderRow - HF_NCC_HEIGHT, blockLeaderCol - HF_NCC_WIDTH, imgHeight, imgWidth);

    float bestSimilarity = -1e5;
    int dispBest = INFTY;
    for (int disparityBlock = MAX_DISP / BLOCK_SIZE; disparityBlock >= 0; disparityBlock--) {
        int dispStart = disparityBlock * BLOCK_SIZE;
        int disparityBlockLeaderCol = blockLeaderCol - dispStart;
        if (cellInvalid || disparityBlockLeaderCol < 0) {
            __syncthreads();
            __syncthreads();
            continue;
        }
        loadIntoBuffer(problem->img2, buffer2, blockLeaderRow - HF_NCC_HEIGHT, disparityBlockLeaderCol - HF_NCC_WIDTH, imgHeight, imgWidth);

        int dispEnd = dispStart - BLOCK_SIZE;
        float similarity;
        for (int disp = dispStart, dispDel = 0; disp > dispEnd; disp--, dispDel++) {
            if (col >= disp && disp + tc >= 0) {
                #if (USE_NCC)
                    similarity = computeNCCShared(buffer1, buffer2, tr + HF_NCC_HEIGHT, tc + HF_NCC_WIDTH, dispDel + HF_NCC_WIDTH);
                #else
                    similarity = computeSSDShared(buffer1, buffer2, tr + HF_NCC_HEIGHT, tc + HF_NCC_WIDTH, dispDel + HF_NCC_WIDTH);
                #endif
                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity;
                    dispBest = disp + tc;
                }
            }
        }
        __syncthreads();
    }

    if (cellInvalid)
        return;

    if (dispBest == INFTY)
        problem->res[row * imgWidth + col] = 0.0;
    else {
        if (dispBest < 0)
            dispBest = 0;
        problem->res[row * imgWidth + col] = dispBest;
    }
}


float* computeDisparityMapShared(Float3* img1, Float3* img2, int height, int width) {
    float* res;
    cudaMalloc(&res, sizeof(float) * height * width);
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, res);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);
    double tStart = clock();

    dim3 threadDim(BLOCK_SIZE, BLOCK_SIZE);
    int blockCountX = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blockCountY = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(blockCountX, blockCountY);
    disparityKernel<<<blockDim, threadDim>>>(problemGPU);
    cudaDeviceSynchronize();

    double tEnd = clock();
    printf("Kernel call took %.2lf ms.\n", (tEnd - tStart) / CLOCKS_PER_SEC * 1000.0);
    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(res);
    cudaFree(problemGPU);
    return resCPU;
}

