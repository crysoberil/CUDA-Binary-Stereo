#include <stdio.h>
#include "libpng_wrapper.h"
#include "stereo_cpu.h"
//#include "stereo_cuda.h"
//#include "stereo_cuda_shared.h"
#include "stereo_cuda_3d.h"



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


void stereoCPU(char* img1Path, char* img2Path, char* resultPath) {
    GrayscaleImage img1, img2;
    readPNGFile(img1, img1Path);
    readPNGFile(img2, img2Path);
    int height = img1.height;
    int width = img1.width;
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
    writePNGFile(resImg, resultPath);
}


float* get_flattened_color_arrayGPU(GrayscaleImage &img) {
    int n = img.height * img.width;
    float* arrCPU = new float[n];
    float* arrGPU;
    cudaMalloc(&arrGPU, sizeof(float) * n);
    int k = 0;
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++, k++)
            arrCPU[k] = img.img[i][j] / 255.0;
    }
    cudaMemcpy(arrGPU, arrCPU, sizeof(float) * n, cudaMemcpyHostToDevice);
    delete[] arrCPU;
    return arrGPU;
}


void stereoGPU(char* img1Path, char* img2Path, char* resultPath) {
    GrayscaleImage img1, img2;
    readPNGFile(img1, img1Path);
    readPNGFile(img2, img2Path);
    int height = img1.height;
    int width = img1.width;
    float* colors1 = get_flattened_color_arrayGPU(img1);
	float* colors2 = get_flattened_color_arrayGPU(img2);
	float* res = computeDisparityMap3D(colors1, colors2, height, width);
    DoubleImage resImg;
	resImg.init(height, width);
	int k = 0;
	for (int i = 0; i < height; i++) {
	    for (int j = 0; j < width; j++, k++)
	        resImg.img[i][j] = res[k];
	}
	cudaFree(colors1);
    cudaFree(colors2);
    delete[] res;
    writePNGFile(resImg, resultPath);
}


void testMiddleBuryCPU() {
    char s1[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small_gr.png";
	char s2[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5_small_gr.png";
	char s3[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_cpu.png";
	stereoCPU(s1, s2, s3);
}


void testMiddleBuryGPU() {
    char s1[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1_small_gr.png";
	char s2[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5_small_gr.png";
	char s3[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_gpu.png";
	stereoGPU(s1, s2, s3);
}


int main() {
//    testMiddleBuryCPU();
    testMiddleBuryGPU();
	return 0;
}
