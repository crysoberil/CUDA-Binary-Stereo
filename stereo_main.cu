#include <stdio.h>
#include "libpng_wrapper.h"
#include "stereo_cpu.h"
//#include "stereo_cuda.h"
//#include "stereo_cuda_shared.h"
#include "stereo_cuda_3d.h"


//void testMiddleBuryCPU() {
//    char s1[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1.png";
//	Image img1;
//	readPNGFile(img1, s1);
//	char s2[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5.png";
//	Image img2;
//	readPNGFile(img2, s2);
//	img1.displayStats();
//	img2.displayStats();
//
//	char s3[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out_cpu.png";
//	BinaryStereoCPU binStereo(&img1, &img2, 7);
//	DoubleImage res;
//	binStereo.computeStereo(res);
//    writePNGFile(res, s3);
//
//	printf("Done\n");
//}


float* get_flattened_color_array(GrayscaleImage &img) {
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
    float* colors1 = get_flattened_color_array(img1);
	float* colors2 = get_flattened_color_array(img2);
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
