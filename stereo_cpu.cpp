#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector_functions.h>
#include "stereo_cpu.h"
#include <math.h>


//#define BLOCK_SIZE 32
#define NCC_HEIGHT 3
#define NCC_WIDTH 7
#define HF_NCC_HEIGHT (NCC_HEIGHT / 2)
#define HF_NCC_WIDTH (NCC_WIDTH / 2)
#define MAX_DISP 128
#define INFTY (1 << 29)


void loadMeanStdCPU(float *img, int rCenter, int cCenter, float &mean, float &invStd, int imgHeight, int imgWidth) {
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

    double varSum, diff;
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
        invStd = 1.0 / sqrt(varSum);
    }
}


float computeNCCCPU(float *img1, float *img2, float* meanInvStdCache, int r, int c1, int c2, int imgHeight, int imgWidth) {
    float* cache1 = meanInvStdCache + (r * imgWidth + c1) * 4;
    float* cache2 = meanInvStdCache + (r * imgWidth + c2) * 4 + 2;

    float mean1 = *cache1;
    float mean2 = *cache2;
    float invStd1 = cache1[1];
    float invStd2 = cache2[1];

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


void computeDisparityNCCCPU(ProblemCPU* problem) {
    int imgHeight = problem->height;
    int imgWidth = problem->width;

    for (int row = 0; row < problem->height; row++) {
        for (int col = 0; col < problem->width; col++) {
            for (int disp = 0; disp < MAX_DISP; disp++) {
                int nccSetIdx = row * imgWidth * MAX_DISP + col * MAX_DISP + disp;
                int rightImgCol = col - disp;
                if (rightImgCol < 0 || rightImgCol >= imgWidth) {
                    problem->nccSet[nccSetIdx] = -INFTY;
                    continue;
                }
                float ncc = computeNCCCPU(problem->img1, problem->img2, problem->meanInvStdCache, row, col, rightImgCol, imgHeight, imgWidth);
                problem->nccSet[nccSetIdx] = ncc;
            }
        }
    }
}


void computeDisparity(ProblemCPU* problem) {
    int imgWidth = problem->width;

    for (int row = 0; row < problem->height; row++) {
        for (int col = 0; col < problem->width; col++) {
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
    }
}


void cacheMeanInvStdCPU(ProblemCPU* problem) {
    int imgWidth = problem->width;

    for (int row = 0; row < problem->height; row++) {
        for (int col = 0; col < problem->width; col++) {
            float *cache = problem->meanInvStdCache + (row * imgWidth + col) * 4;
            float mean1, invstd1, mean2, invstd2;
            loadMeanStdCPU(problem->img1, row, col, mean1, invstd1, problem->height, imgWidth);
            loadMeanStdCPU(problem->img2, row, col, mean2, invstd2, problem->height, imgWidth);
            *cache = mean1;
            cache[1] = invstd1;
            cache[2] = mean2;
            cache[3] = invstd2;
        }
    }
}


float* computeDisparityMapCPU(float* img1, float* img2, int height, int width) {
    float* res = new float[height * width];
    float* nccSet = new float[height * width * MAX_DISP];
    float* meanInvStdCache = new float[height * width * 4];
    ProblemCPU problem(img1, img2, height, width, nccSet, meanInvStdCache, res);

    // Kernel 1 - sequential
    cacheMeanInvStdCPU(&problem);

    // Kernel 2 - sequential
    computeDisparityNCCCPU(&problem);

    // Kernel 3 - sequential
    computeDisparity(&problem);

    delete[] nccSet;
    delete[] meanInvStdCache;

    return res;
}
