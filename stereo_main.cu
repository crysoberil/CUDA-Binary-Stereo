#include <stdio.h>
#include "libpng_wrapper.h"
#include "stereo_cpu.h"
#include "stereo_cuda.h"
#include "stereo_cuda_shared.h"


void testMiddleBuryCPU() {
    char s1[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1.png";
	Image img1;
	readPNGFile(img1, s1);
	char s2[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5.png";
	Image img2;
	readPNGFile(img2, s2);
	img1.displayStats();
	img2.displayStats();

	char s3[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_cpu.png";
	BinaryStereoCPU binStereo(&img1, &img2, 7);
	DoubleImage res;
	binStereo.computeStereo(res);
    writePNGFile(res, s3);

	printf("Done\n");
}


unsigned char* get_flattened_color_array(int height, int width) {
    unsigned char* arr;
    int n = height * width * 3;
    cudaMallocManaged(&arr, n * sizeof(unsigned char));
    return arr;
}


unsigned char* get_flattened_color_array(Image &img) {
    int n = img.height * img.width * 3;
    unsigned char* arrCPU = new unsigned char[n];
    unsigned char* arrGPU;
    cudaMalloc(&arrGPU, sizeof(unsigned char) * n);
    int k = 0;
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            for (int channel = 0; channel < 3; channel++)
                arrCPU[k++] = img.img[i][j][channel];
        }
    }
    cudaMemcpy(arrGPU, arrCPU, sizeof(unsigned char) * n, cudaMemcpyHostToDevice);
    delete[] arrCPU;
    return arrGPU;
}


void stereoGPU(char* img1Path, char* img2Path, char* resultPath) {
    Image img1, img2;
    readPNGFile(img1, img1Path);
    readPNGFile(img2, img2Path);
    int height = img1.height;
    int width = img1.width;
    unsigned char* colors1 = get_flattened_color_array(img1);
	unsigned char* colors2 = get_flattened_color_array(img2);
//	float* res = computeDisparityMap(colors1, colors2, height, width, 7);
	float* res = computeDisparityMapShared(colors1, colors2, height, width);
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



void testMiddleBuryGPU() {
    char s1[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1.png";
	char s2[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5.png";
	char s3[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_gpu.png";
	stereoGPU(s1, s2, s3);
}


int main() {
//    testMiddleBuryCPU();
    testMiddleBuryGPU();
	return 0;
}
