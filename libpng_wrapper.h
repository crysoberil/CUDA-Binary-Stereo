#ifndef PNG_HEADER_H
#define PNG_HEADER_H


#include <stdio.h>


class Image {
    void clearMemory() {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++)
                delete[] img[i][j];
            delete[] img[i];
        }
	    if (height > 0)
	        delete[] img;
    }

public:
    unsigned char ***img;
    int height, width;


    void init(int height, int width) {
        this->height = height;
	    this->width = width;
	    if (this->height <= 0 or this->width <= 0)
	        return;
	    img = new unsigned char**[height];
	    for(int i = 0; i < height; i++) {
	        img[i] = new unsigned char*[width];
	        for (int j = 0; j < width; j++)
	            img[i][j] = new unsigned char[3];
	    }
    }

    Image() {
        init(-1, -1);
    }


	Image(int height, int width) {
	    init(height, width);
	}

	~Image() {
	    clearMemory();
	}

	void loadColoredPixel(int row, int col, unsigned char r, unsigned char g, unsigned char b) {
	    img[row][col][0] = r;
	    img[row][col][1] = g;
	    img[row][col][2] = b;
	}

	void displayStats() {
	    printf("Image shape=(%d, %d)\n", height, width);
	}
};


class DoubleImage {
    void clearMemory() {
        for (int i = 0; i < height; i++) {
            delete[] img[i];
        }
	    if (height > 0)
	        delete[] img;
    }

public:
    double **img;
    int height, width;


    void init(int height, int width) {
        this->height = height;
	    this->width = width;
	    if (this->height <= 0 or this->width <= 0)
	        return;
	    img = new double*[height];
	    for(int i = 0; i < height; i++) {
	        img[i] = new double[width];
	        for (int j = 0; j < width; j++)
	            img[i][j] = 0.0;
	    }
    }

    DoubleImage() {
        init(-1, -1);
    }


	DoubleImage(int height, int width) {
	    init(height, width);
	}

	~DoubleImage() {
	    clearMemory();
	}

	void initFrom(DoubleImage &sec) {
	    init(sec.height, sec.width);
	    for (int i = 0; i < height; i++) {
	        for (int j = 0; j < width; j++)
	            img[i][j] = sec.img[i][j];
	    }
	}

	void scaleByMax(double scale) {
	    double maxm = -1e10;
	    for (int i = 0; i < height; i++) {
	        for (int j = 0; j < width; j++) {
	            if (img[i][j] > maxm)
	                maxm = img[i][j];
	        }
	    }

        double mult = scale / maxm;

	    for (int i = 0; i < height; i++) {
	        for (int j = 0; j < width; j++)
	            img[i][j] *= mult;
	    }
	}


	void scaleBy(double f) {
	    for (int i = 0; i < height; i++) {
	        for (int j = 0; j < width; j++) {
	            img[i][j] *= f;
	        }
	    }
	}


	void displayStats() {
	    printf("Image shape=(%d, %d)\n", height, width);
	}
};


void readPNGFile(Image &img, char* fPath);

void writePNGFile(DoubleImage &img, char* fPath);


#endif