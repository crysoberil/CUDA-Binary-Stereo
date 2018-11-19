#include <math.h>
#include <stdio.h>
#include "stereo_cuda_shared.h"


#define SHARED_MEM_BLOCK_SIZE 32
#define NCC_WINDOW_HEIGHT 3
#define NCC_WINDOW_WIDTH 7
#define COLOR_PATCH_HEIGHT (SHARED_MEM_BLOCK_SIZE + NCC_WINDOW_HEIGHT - 1)
#define COLOR_PATCH_WIDTH (SHARED_MEM_BLOCK_SIZE + NCC_WINDOW_WIDTH - 1)

#define NUM_COLOR_CHANNELS 3


typedef union {
    float4 f;
    float arr[4];
} Float4;


struct Problem {
public:
    unsigned char* img1;
    unsigned char* img2;
    int height, width;
    float* res;

    Problem(unsigned char* img1, unsigned char* img2, int height, int width, float* res) {
        this->img1 = img1;
        this->img2 = img2;
        this->height = height;
        this->width = width;
        this->res = res;
    }
};


__device__
void getWindowMeanSTDShared(unsigned char colorArr[COLOR_PATCH_HEIGHT][COLOR_PATCH_WIDTH][NUM_COLOR_CHANNELS], int centerRow, int centerCol, Float4 &mean, Float4 &std) {
    Float4 windowSum;
    windowSum.f.x = windowSum.f.y = windowSum.f.z = 0.0;
    int windowSize = 0;

    int halfWindowRow = NCC_WINDOW_HEIGHT / 2;
    int halfWindowCol = NCC_WINDOW_WIDTH / 2;

    for (int r = centerRow - halfWindowRow; r <= centerRow + halfWindowRow; r++) {
        for (int c = centerCol - halfWindowCol; c <= centerCol + halfWindowCol; c++) {
            if (r >= 0 && r < COLOR_PATCH_HEIGHT && c >= 0 && c < COLOR_PATCH_WIDTH) {
                for (int channel = 0; channel < 3; channel++) {
                    windowSum.arr[channel] += colorArr[r][c][channel];
                }
                windowSize++;
            }
        }
    }

    // Compute average
    for (int channel = 0; channel < NUM_COLOR_CHANNELS; channel++) {
        mean.arr[channel] = windowSum.arr[channel] / windowSize;
    }

    Float4 varSum;
    varSum.f.x = varSum.f.y = varSum.f.z = 0.0;

    for (int r = centerRow - halfWindowRow; r <= centerRow + halfWindowRow; r++) {
        for (int c = centerCol - halfWindowCol; c <= centerCol + halfWindowCol; c++) {
            if (r >= 0 && r < COLOR_PATCH_HEIGHT && c >= 0 && c < COLOR_PATCH_WIDTH) {
                for (int channel = 0; channel < 3; channel++) {
                    float diff = colorArr[r][c][channel] - mean.arr[channel];
                    varSum.arr[channel] += diff * diff;
                }
            }
        }
    }

    for (int channel = 0; channel < 3; channel++) {
        std.arr[channel] = sqrt(varSum.arr[channel] / windowSize);
        if (std.arr[channel] < 1e-4)
            std.arr[channel] = 1e-4;
    }
}


__device__
float computeNCCSharedMem(unsigned char img1[COLOR_PATCH_HEIGHT][COLOR_PATCH_WIDTH][NUM_COLOR_CHANNELS], int row1, int col1,
            unsigned char img2[COLOR_PATCH_HEIGHT][COLOR_PATCH_WIDTH][NUM_COLOR_CHANNELS], int row2, int col2) {
    float ncc = 0.0;
    Float4 mean1;
    Float4 std1;
    Float4 mean2;
    Float4 std2;
    getWindowMeanSTDShared(img1, row1, col1, mean1, std1);
    getWindowMeanSTDShared(img2, row2, col2, mean2, std2);
    int totalContribCount = 0;
    int halfWindowRow = NCC_WINDOW_HEIGHT / 2;
    int halfWindowCol = NCC_WINDOW_WIDTH / 2;
    for (int rDel = -halfWindowRow; rDel <= halfWindowRow; rDel++) {
        for (int cDel = -halfWindowCol; cDel <= halfWindowCol; cDel++) {
            int r1 = row1 + rDel;
            int c1 = col1 + cDel;
            int r2 = row2 + rDel;
            int c2 = col2 + cDel;
            if (r1 >= 0 && r1 < COLOR_PATCH_HEIGHT && c1 >= 0 && c1 < COLOR_PATCH_WIDTH && r2 >= 0 && r2 < COLOR_PATCH_HEIGHT && c2 >= 0 && c2 < COLOR_PATCH_WIDTH) {
                for (int channel = 0; channel < NUM_COLOR_CHANNELS; channel++) {
                    float contrib = (img1[r1][c1][channel] - mean1.arr[channel]) * (img2[r2][c2][channel] - mean2.arr[channel]) / (std1.arr[channel] * std2.arr[channel]);
                    ncc += contrib / NUM_COLOR_CHANNELS;  // To account for 3 channels.
                }
                totalContribCount++;
            }
        }
    }
    float avg = ncc / totalContribCount;
    return avg;
}


__global__
void computeDisparityKernelShared(Problem* problem) {
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    int imgHeight = problem->height;
    int imgWidth = problem->width;

    int blockHeadRow = SHARED_MEM_BLOCK_SIZE * by;
    int blockHeadCol = SHARED_MEM_BLOCK_SIZE * bx;
    int row = blockHeadRow + ty;
    int col = blockHeadCol + tx;

    if (row >= imgHeight || col >= imgWidth)
        return;

    __shared__ unsigned char color1Buffer[COLOR_PATCH_HEIGHT][COLOR_PATCH_WIDTH][NUM_COLOR_CHANNELS];
    __shared__ unsigned char color2Buffer[COLOR_PATCH_HEIGHT][COLOR_PATCH_WIDTH][NUM_COLOR_CHANNELS];

    // Now load into the shared memory.
    int responsibilityAccrossRow = 2 + (NCC_WINDOW_HEIGHT - 2) / SHARED_MEM_BLOCK_SIZE;
    int responsibilityAccrossCol = 2 + (NCC_WINDOW_WIDTH - 2) / SHARED_MEM_BLOCK_SIZE;

    for (int i = 0; i < responsibilityAccrossRow; i++) {
        int sharedIdxRow = responsibilityAccrossRow * ty + i;
        int img1Row = sharedIdxRow + blockHeadRow - NCC_WINDOW_HEIGHT / 2;
        if (img1Row < 0 || img1Row >= imgHeight || sharedIdxRow >= SHARED_MEM_BLOCK_SIZE)
            continue;
        for (int j = 0; j < responsibilityAccrossCol; j++) {
            int sharedIdxCol = responsibilityAccrossCol * tx + j;
            int img1Col = sharedIdxCol + blockHeadCol - NCC_WINDOW_WIDTH / 2;
            if (img1Col < 0 || img1Col >= imgWidth || sharedIdxCol >= SHARED_MEM_BLOCK_SIZE)
                continue;
            for (int channel = 0; channel < NUM_COLOR_CHANNELS; channel++) {
                color1Buffer[sharedIdxRow][sharedIdxCol][channel] = problem->img1[img1Row * imgWidth * 3 + img1Col * 3 + channel];
            }
        }
    }
    __syncthreads();

    float bestNCC = -1e10;
    int bestMatchedColumn;

    int columnwiseChunks = (imgWidth + SHARED_MEM_BLOCK_SIZE - 1) / SHARED_MEM_BLOCK_SIZE;
    for (int columnwiseChunk = 0; columnwiseChunk < columnwiseChunks; columnwiseChunk++) {
        int secondBlockHeadCol = SHARED_MEM_BLOCK_SIZE * columnwiseChunk;
        for (int i = 0; i < responsibilityAccrossRow; i++) {
            int sharedIdxRow = responsibilityAccrossRow * ty + i;
            int img2Row = sharedIdxRow + blockHeadRow - NCC_WINDOW_HEIGHT / 2;
            if (img2Row < 0 || img2Row >= imgHeight || sharedIdxRow >= SHARED_MEM_BLOCK_SIZE)
                continue;
            for (int j = 0; j < responsibilityAccrossCol; j++) {
                int sharedIdxCol = responsibilityAccrossCol * tx + j;
                int img2Col = sharedIdxCol + secondBlockHeadCol - NCC_WINDOW_WIDTH / 2;
                if (img2Col < 0 || img2Col >= imgWidth || sharedIdxCol >= SHARED_MEM_BLOCK_SIZE)
                    continue;
                for (int channel = 0; channel < NUM_COLOR_CHANNELS; channel++) {
                    color2Buffer[sharedIdxRow][sharedIdxCol][channel] = problem->img2[img2Row * imgWidth * 3 + img2Col * 3 + channel];
                }
            }
        }
        __syncthreads();

        for (int colSecShared = 0; colSecShared < SHARED_MEM_BLOCK_SIZE && colSecShared + columnwiseChunk * SHARED_MEM_BLOCK_SIZE < imgWidth; colSecShared++) {
            float ncc = computeNCCSharedMem(color1Buffer, ty + NCC_WINDOW_HEIGHT / 2, tx + NCC_WINDOW_WIDTH / 2, color2Buffer, ty + NCC_WINDOW_HEIGHT / 2, colSecShared + NCC_WINDOW_WIDTH / 2);
            if (ncc > bestNCC) {
                bestNCC = ncc;
                bestMatchedColumn = colSecShared + columnwiseChunk * SHARED_MEM_BLOCK_SIZE;
            }
        }
    }

    int colDiff = bestMatchedColumn - col;
    if (colDiff < 0)
        colDiff = -colDiff;
    problem->res[problem->width * row + col] = colDiff;
}


float* computeDisparityMapShared(unsigned char* img1, unsigned char* img2, int height, int width) {
    float* res;
    cudaMalloc(&res, sizeof(float) * height * width);
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, res);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);

    int numOfCells = height * width;
    int threadDim = SHARED_MEM_BLOCK_SIZE;
    dim3 threadDimension(threadDim, threadDim);
    int numBlocks = (numOfCells + threadDim * threadDim - 1) / (threadDim * threadDim);
    int blockCountX = (int)(sqrt(1.0 * numBlocks));
    int blockCountY = (numBlocks + blockCountX - 1) / blockCountX;
    dim3 blockDimension(blockCountX, blockCountY);
    computeDisparityKernelShared<<<blockDimension, threadDimension>>>(problemGPU);

    cudaDeviceSynchronize();
    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(res);
    cudaFree(problemGPU);
    return resCPU;
}

