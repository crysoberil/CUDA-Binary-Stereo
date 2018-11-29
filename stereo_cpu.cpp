//#include <math.h>
//#include <stdio.h>
//#include "stereo_cpu.h"
//
//
//void BinaryStereoCPU::computeStereo(DoubleImage &resultContainer) {
//    resultContainer.init(img1->height, img1->width);
//    for (int i = 0; i < resultContainer.height; i++) {
//        printf("Processing row: %d\n", i);
//        for (int j = 0; j < resultContainer.width; j++) {
//            int bestMatchedCol = findBestMatchingColumnInSecondImage(i, j);
//            int disp = j - bestMatchedCol;
//            if (disp < 0)
//                disp = -disp;
//            resultContainer.img[i][j] = disp;
//        }
//    }
//}
//
//
//int BinaryStereoCPU::findBestMatchingColumnInSecondImage(int row, int col1) {
//    double bestNCC = -1e10;
//    int bestCol2;
//
//    for (int col2 = 0; col2 < img2->width; col2++) {
//        double ncc = computeNCC(row, col1, row, col2);
//        if (ncc > bestNCC) {
//            bestNCC = ncc;
//            bestCol2 = col2;
//        }
//    }
//
//    return bestCol2;
//}
//
//
//double BinaryStereoCPU::computeNCC(int row1, int col1, int row2, int col2) {
//    double ncc = 0.0;
//    double mean1[3];
//    double std1[3];
//    double mean2[3];
//    double std2[3];
//    getWindowMeanSTD(img1, row1, col1, mean1, std1);
//    getWindowMeanSTD(img2, row2, col2, mean2, std2);
//    int halfWindow = nccWindowSize / 2;
//    int totalContribCount = 0;
//    for (int rDel = -halfWindow; rDel <= halfWindow; rDel++) {
//        for (int cDel = -halfWindow; cDel <= halfWindow; cDel++) {
//            int r1 = row1 + rDel;
//            int c1 = col1 + cDel;
//            int r2 = row2 + rDel;
//            int c2 = col2 + cDel;
//            if (r1 >= 0 && r1 < img1->height && c1 >= 0 && c1 < img1->width && r2 >= 0 && r2 < img2->height && c2 >= 0 && c2 < img2->width) {
//                for (int channel = 0; channel < 3; channel++) {
//                    double contrib = (img1->img[r1][c1][channel] - mean1[channel]) * (img2->img[r2][c2][channel] - mean2[channel]) / (std1[channel] * std2[channel]);
//                    ncc += contrib / 3.0;  // To account for 3 channels.
//                }
//                totalContribCount++;
//            }
//        }
//    }
//    double avg = ncc / totalContribCount;
//    return avg;
//}
//
//
//void BinaryStereoCPU::getWindowMeanSTD(Image *img, int centerRow, int centerCol, double *mean, double *std) {
//    double windowSum[3] = {0.0, 0.0, 0.0};
//    int windowSize = 0;
//
//    int halfWindow = nccWindowSize / 2;
//    for (int r = centerRow - halfWindow; r <= centerRow + halfWindow; r++) {
//        for (int c = centerCol - halfWindow; c <= centerCol + halfWindow; c++) {
//            if (r >= 0 && r < img->height && c >= 0 && c < img->width) {
//                for (int channel = 0; channel < 3; channel++) {
//                    windowSum[channel] += img->img[r][c][channel];
//                }
//                windowSize++;
//            }
//        }
//    }
//
//    // Compute average
//    for (int channel = 0; channel < 3; channel++) {
//        mean[channel] = windowSum[channel] / windowSize;
//    }
//
//    double varSum[3] = {0.0, 0.0, 0.0};
//
//    for (int r = centerRow - halfWindow; r <= centerRow + halfWindow; r++) {
//        for (int c = centerCol - halfWindow; c <= centerCol + halfWindow; c++) {
//            if (r >= 0 && r < img->height && c >= 0 && c < img->width) {
//                for (int channel = 0; channel < 3; channel++) {
//                    double diff = img->img[r][c][channel] - mean[channel];
//                    varSum[channel] += diff * diff;
//                }
//            }
//        }
//    }
//
//    for (int channel = 0; channel < 3; channel++) {
//        std[channel] = sqrt(varSum[channel] / windowSize);
//        if (std[channel] < 1e-5)
//            std[channel] = 1e-5;
//    }
//}