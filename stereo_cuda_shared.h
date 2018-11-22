#ifndef STEREO_GPU_SHARED_HEADER_H
#define STEREO_GPU_SHARED_HEADER_H


typedef union {
    float3 f;
    float arr[3];
} Float3;


//class __align__(16) Float3_ {
//public:
//    float x, y, z;
//    int __alignmentPlaceholder__;
//
//    __host__ __device__
//    Float3_() {
//        x = 0;
//        y = 0;
//        z = 0;
//    }
//
//    __host__ __device__
//    Float3_(float x, float y, float z) {
//        this->x = x;
//        this->y = y;
//        this->z = z;
//    }
//
//    __host__ __device__
//    float& operator[](int idx) {
//        switch (idx) {
//            case 0:
//                return x;
//            case 1:
//                return y;
//            case 2:
//                return z;
//        }
//        return x;
//    }
//};


struct Problem {
public:
    Float3* img1;
    Float3* img2;
    int height, width;
    float* res;

    Problem(Float3* img1, Float3* img2, int height, int width, float* res) {
        this->img1 = img1;
        this->img2 = img2;
        this->height = height;
        this->width = width;
        this->res = res;
    }
};


float* computeDisparityMapShared(Float3* img1, Float3* img2, int height, int width);


#endif