#include "stereo_cuda.h"
#include <stdio.h>


struct Problem {
public:
    unsigned char* img1;
    unsigned char* img2;
    int height, width, nccWindowSize;
    float* res;

    Problem(unsigned char* img1, unsigned char* img2, int height, int width, float* res, int nccWindowSize) {
        this->img1 = img1;
        this->img2 = img2;
        this->height = height;
        this->width = width;
        this->res = res;
        this->nccWindowSize = nccWindowSize;
    }
};


__device__
void getWindowMeanSTD(Problem* problem, unsigned char* colorArr, int centerRow, int centerCol, float *mean, float *std) {
    float windowSum[3] = {0.0, 0.0, 0.0};
    int windowSize = 0;

    int halfWindow = problem->nccWindowSize / 2;
    int colorArrIdx;

    for (int r = centerRow - halfWindow; r <= centerRow + halfWindow; r++) {
        for (int c = centerCol - halfWindow; c <= centerCol + halfWindow; c++) {
            if (r >= 0 && r < problem->height && c >= 0 && c < problem->width) {
                for (int channel = 0; channel < 3; channel++) {
                    colorArrIdx = r * problem->width * 3 + c * 3 + channel;
                    windowSum[channel] += colorArr[colorArrIdx];
                }
                windowSize++;
            }
        }
    }

    // Compute average
    for (int channel = 0; channel < 3; channel++) {
        mean[channel] = windowSum[channel] / windowSize;
    }

    float varSum[3] = {0.0, 0.0, 0.0};

    for (int r = centerRow - halfWindow; r <= centerRow + halfWindow; r++) {
        for (int c = centerCol - halfWindow; c <= centerCol + halfWindow; c++) {
            if (r >= 0 && r < problem->height && c >= 0 && c < problem->width) {
                for (int channel = 0; channel < 3; channel++) {
                    colorArrIdx = r * problem->width * 3 + c * 3 + channel;
                    float diff = colorArr[colorArrIdx] - mean[channel];
                    varSum[channel] += diff * diff;
                }
            }
        }
    }

    for (int channel = 0; channel < 3; channel++) {
        std[channel] = sqrt(varSum[channel] / windowSize);
        if (std[channel] < 1e-4)
            std[channel] = 1e-4;
    }
}


__device__
float computeNCC(Problem *problem, int row1, int col1, int row2, int col2) {
    float ncc = 0.0;
    float mean1[3];
    float std1[3];
    float mean2[3];
    float std2[3];
    getWindowMeanSTD(problem, problem->img1, row1, col1, mean1, std1);
    getWindowMeanSTD(problem, problem->img2, row2, col2, mean2, std2);
    int halfWindow = problem->nccWindowSize / 2;
    int totalContribCount = 0;
    for (int rDel = -halfWindow; rDel <= halfWindow; rDel++) {
        for (int cDel = -halfWindow; cDel <= halfWindow; cDel++) {
            int r1 = row1 + rDel;
            int c1 = col1 + cDel;
            int r2 = row2 + rDel;
            int c2 = col2 + cDel;
            if (r1 >= 0 && r1 < problem->height && c1 >= 0 && c1 < problem->width && r2 >= 0 && r2 < problem->height && c2 >= 0 && c2 < problem->width) {
                for (int channel = 0; channel < 3; channel++) {
                    int img1Idx = r1 * problem->width * 3 + c1 * 3 + channel;
                    int img2Idx = r2 * problem->width * 3 + c2 * 3 + channel;
                    float contrib = (problem->img1[img1Idx] - mean1[channel]) * (problem->img2[img2Idx] - mean2[channel]) / (std1[channel] * std2[channel]);
                    ncc += contrib / 3.0;  // To account for 3 channels.
                }
                totalContribCount++;
            }
        }
    }
    float avg = ncc / totalContribCount;
    return avg;
}


__global__
void computeDisparityKernel(Problem* problem) {
    int linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIdx >= problem->height * problem->width)
        return;

    int row = linearIdx / problem->width;
    int col = linearIdx % problem->width;

    float bestNCC = -1e10;
    int bestColSec;

    for (int colSec = 0; colSec < problem->width; colSec++) {
        float ncc = computeNCC(problem, row, col, row, colSec);
        if (ncc > bestNCC) {
            bestNCC = ncc;
            bestColSec = colSec;
        }
    }

    int colDiff = bestColSec - col;
    if (colDiff < 0)
        colDiff = -colDiff;

    problem->res[problem->width * row + col] = colDiff;
}


float* computeDisparityMap(unsigned char* img1, unsigned char* img2, int height, int width, int nccWindowSize) {
    float* res;
    cudaMalloc(&res, sizeof(float) * height * width);
    Problem* problemGPU;
    Problem problemCPU(img1, img2, height, width, res, nccWindowSize);
    cudaMalloc(&problemGPU, sizeof(Problem));
    cudaMemcpy(problemGPU, &problemCPU, sizeof(Problem), cudaMemcpyHostToDevice);
    int numOfCells = height * width;
    int numThreads = 512;
    int numBlocks = (numOfCells + numThreads - 1) / numThreads;
    computeDisparityKernel<<<numBlocks, numThreads>>>(problemGPU);
    cudaDeviceSynchronize();
    float* resCPU = new float[height * width];
    cudaMemcpy(resCPU, res, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
    cudaFree(res);
    cudaFree(problemGPU);
    return resCPU;
}