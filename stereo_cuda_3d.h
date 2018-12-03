#ifndef STEREO_GPU_3D_HEADER_H
#define STEREO_GPU_3D_HEADER_H


struct Problem {
public:
    float* img1;
    float* img2;
    int height, width;
    float* nccSet;
    float* res;
    float* meanInvStdCache;

    Problem(float* img1, float* img2, int height, int width, float* nccSet, float* meanInvStdCache, float* res) {
        this->img1 = img1;
        this->img2 = img2;
        this->height = height;
        this->width = width;
        this->nccSet = nccSet;
        this->meanInvStdCache = meanInvStdCache;
        this->res = res;
    }
};


float* computeDisparityMap3D(float* img1, float* img2, int height, int width, float* meanInvStdCache, float* nccSet, float* res);

#endif