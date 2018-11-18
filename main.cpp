#include <stdio.h>
#include "libpng_wrapper.h"
#include "stereo_cpu.h"


void testMiddleBury() {
    char s1[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view1.png";
	Image img1;
	readPNGFile(img1, s1);
	char s2[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/view5.png";
	Image img2;
	readPNGFile(img2, s2);
	img1.displayStats();
	img2.displayStats();

	char s3[] = "/playpen2/jisan/workspace/Datasets/Middlebury/Art/out.png";
	BinaryStereoCPU binStereo(&img1, &img2, 7);
	DoubleImage res;
	binStereo.computeStereo(res);
    writePNGFile(res, s3);

	printf("Done\n");
}


int main() {
	testMiddleBury();
	return 0;
}
