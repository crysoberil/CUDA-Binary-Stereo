#include <stdio.h>
#include "libpng_wrapper.h"
#include "stereo_cpu.h"
//#include "stereo_cuda.h"
//#include "stereo_cuda_shared.h"
#include "stereo_cuda_3d.h"
#include <time.h>



float* get_flattened_color_arrayCPU(GrayscaleImage &img) {
    int n = img.height * img.width;
    float* arrCPU = new float[n];
    int k = 0;
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++, k++)
            arrCPU[k] = img.img[i][j] / 255.0;
    }
    return arrCPU;
}


void stereoCPU(char* img1Path, char* img2Path, char* resultPath, int testCases, bool isDummy, bool writeToFile) {
    GrayscaleImage img1, img2;
    readPNGFile(img1, img1Path);
    readPNGFile(img2, img2Path);
    int height = img1.height;
    int width = img1.width;

    int maxDisp = 128;
    int windowHeight = 3;
    int windowWidth = 7;
    int n = img1.height * img1.width;

    double clockStart = clock();

    for (int tCase = 1; tCase <= testCases; tCase++) {
        float* colors1 = get_flattened_color_arrayCPU(img1);
        float* colors2 = get_flattened_color_arrayCPU(img2);
        float* res = computeDisparityMapCPU(colors1, colors2, height, width);
        DoubleImage resImg;
        resImg.init(height, width);
        int k = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++, k++)
                resImg.img[i][j] = res[k];
        }
        delete[] colors1;
        delete[] colors2;
        delete[] res;

        if (writeToFile && tCase == testCases)
            writePNGFile(resImg, resultPath);
    }

    if (isDummy)
        return;

    double clockEnd = clock();
    double exectimeSec = (clockEnd - clockStart) / CLOCKS_PER_SEC;
    printf("Average execution time: %lfms\n", exectimeSec * 1000.0 / testCases);

    long long totalInstructions = ((long long)(n * testCases)) * (maxDisp * windowHeight * windowWidth);
    printf("MIPS obtained: %lf\n", totalInstructions / 1e6 / exectimeSec);
}


void get_flattened_color_arrayGPU(GrayscaleImage &img, float* arrGPU) {
    int n = img.height * img.width;
    float* arrCPU = new float[n];
    int k = 0;
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++, k++)
            arrCPU[k] = img.img[i][j] / 255.0;
    }
    cudaMemcpy(arrGPU, arrCPU, sizeof(float) * n, cudaMemcpyHostToDevice);
    delete[] arrCPU;
}


void stereoGPU(char* img1Path, char* img2Path, char* resultPath, int testCases, bool isDummy, bool writeToFile) {
    GrayscaleImage img1, img2;
    readPNGFile(img1, img1Path);
    readPNGFile(img2, img2Path);
    int height = img1.height;
    int width = img1.width;

    int maxDisp = 128;
    int windowHeight = 3;
    int windowWidth = 7;

    double clockStart = clock();

    float* colors1;
    float* colors2;
    float* nccSet;
    float* meanInvStdCache;
    float* resGPU;
    int n = img1.height * img1.width;
    cudaMalloc(&colors1, sizeof(int) * n);
    cudaMalloc(&colors2, sizeof(int) * n);
    cudaMalloc(&resGPU, sizeof(float) * n);
    cudaMalloc(&nccSet, sizeof(float) * n * maxDisp);
    cudaMalloc(&meanInvStdCache, sizeof(float) * n * 4);

    for (int tCase = 1; tCase <= testCases; tCase++) {
        get_flattened_color_arrayGPU(img1, colors1);
        get_flattened_color_arrayGPU(img2, colors2);
        float* res = computeDisparityMap3D(colors1, colors2, height, width, meanInvStdCache, nccSet, resGPU);
        DoubleImage resImg;
        resImg.init(height, width);
        int k = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++, k++)
                resImg.img[i][j] = res[k];
        }

        delete[] res;
        if (writeToFile && tCase == testCases)
            writePNGFile(resImg, resultPath);
    }

    cudaFree(colors1);
    cudaFree(colors2);
    cudaFree(meanInvStdCache);
    cudaFree(nccSet);
    cudaFree(resGPU);

    if (isDummy)
        return;

    double clockEnd = clock();
    double exectimeSec = (clockEnd - clockStart) / CLOCKS_PER_SEC;
    printf("Average execution time: %lfms\n", exectimeSec * 1000.0 / testCases);

    long long totalInstructions = ((long long)(n * testCases)) * (maxDisp * windowHeight * windowWidth);
    printf("MIPS obtained: %lf\n", totalInstructions / 1e6 / exectimeSec);
}


void testMiddleBuryCPU() {
    char s1[] = "./resources/img_left.png";
	char s2[] = "./resources/img_right.png";
	char s3[] = "./resources/out_cpu.png";
	stereoCPU(s1, s2, s3, 1, true, false);
	stereoCPU(s1, s2, s3, 20, false, true);
}


void testMiddleBuryGPU() {
    char s1[] = "./resources/img_left.png";
	char s2[] = "./resources/img_right.png";
	char s3[] = "./resources/out_gpu.png";
	stereoGPU(s1, s2, s3, 1, true, false);
	stereoGPU(s1, s2, s3, 1000, false, true);
}


int main() {
//    testMiddleBuryCPU();
    testMiddleBuryGPU();
	return 0;
}
