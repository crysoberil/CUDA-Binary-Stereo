//#ifndef STEREO_CPU_HEADER_H
//#define STEREO_CPU_HEADER_H
//
//#include "libpng_wrapper.h"
//
//
//class BinaryStereoCPU {
//    Image* img1;
//    Image* img2;
//    int nccWindowSize;
//
//    int findBestMatchingColumnInSecondImage(int row, int col1);
//    double computeNCC(int row1, int col1, int row2, int col2);
//    void getWindowMeanSTD(Image *img, int centerRow, int centerCol, double *mean, double *std);
//
//public:
//    BinaryStereoCPU(Image* img1, Image* img2, int nccWindowSize) {
//        this->img1 = img1;
//        this->img2 = img2;
//        this->nccWindowSize = nccWindowSize;
//    }
//
//    void computeStereo(DoubleImage &resultContainer);
//};
//
//
//#endif