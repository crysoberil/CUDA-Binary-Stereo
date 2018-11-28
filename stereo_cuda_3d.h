#ifndef STEREO_GPU_3D_HEADER_H
#define STEREO_GPU_3D_HEADER_H


typedef union {
    float3 f;
    float arr[3];
} Float3;


struct Problem {
public:
    Float3* img1;
    Float3* img2;
    int height, width;
    float* nccSet;
    float* res;

    Problem(Float3* img1, Float3* img2, int height, int width, float* nccSet, float* res) {
        this->img1 = img1;
        this->img2 = img2;
        this->height = height;
        this->width = width;
        this->nccSet = nccSet;
        this->res = res;
    }
};


float* computeDisparityMap3D(Float3* img1, Float3* img2, int height, int width);

#endif