#ifndef STEREO_GPU_HEADER_H
#define STEREO_GPU_HEADER_H





float* computeDisparityMapCUDA(unsigned char* img1, unsigned char* img2, int height, int width, int nccWindowSize);


#endif
