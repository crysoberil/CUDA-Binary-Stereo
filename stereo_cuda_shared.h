//#ifndef STEREO_GPU_SHARED_HEADER_H
//#define STEREO_GPU_SHARED_HEADER_H
//#include "float3_lib.h"
//
//
//typedef union {
//    float3 f;
//    float arr[3];
//} Float3;
//
//
//struct Problem {
//public:
//    Float3* img1;
//    Float3* img2;
//    int height, width;
//    float* res;
//
//    Problem(Float3* img1, Float3* img2, int height, int width, float* res) {
//        this->img1 = img1;
//        this->img2 = img2;
//        this->height = height;
//        this->width = width;
//        this->res = res;
//    }
//};
//
//
//float* computeDisparityMapShared(Float3* img1, Float3* img2, int height, int width);
//
//
//#endif