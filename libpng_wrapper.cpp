#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include "libpng_wrapper.h"

png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;

void readPNGFile(Image &img, char* fPath) {
	FILE *fp = fopen(fPath, "rb");

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL,
			NULL);
	if (!png)
		abort();

	png_infop info = png_create_info_struct(png);
	if (!info)
		abort();

	if (setjmp(png_jmpbuf(png)))
		abort();

	png_init_io(png, fp);

	png_read_info(png, info);

	int width = png_get_image_width(png, info);
	int height = png_get_image_height(png, info);
	img.init(height, width);
	color_type = png_get_color_type(png, info);
	bit_depth = png_get_bit_depth(png, info);

	// Read any color_type into 8bit depth, RGBA format.
	// See http://www.libpng.org/pub/png/libpng-manual.txt

	if (bit_depth == 16)
		png_set_strip_16(png);

	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png);

	// PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png);

	if (png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);

	// These color_type don't have an alpha channel then fill it with 0xff.
	if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY
			|| color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

	if (color_type == PNG_COLOR_TYPE_GRAY
			|| color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		png_set_gray_to_rgb(png);

	png_read_update_info(png, info);

	row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
	for (int y = 0; y < height; y++) {
		row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png, info));
	}

	png_read_image(png, row_pointers);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char r = row_pointers[i][j << 2];
			unsigned char g = row_pointers[i][(j << 2) + 1];
			unsigned char b = row_pointers[i][(j << 2) + 2];
			img.loadColoredPixel(i, j, r, g, b);
		}
		free(row_pointers[i]);
	}

	free(row_pointers);
	fclose(fp);
}

void writePNGFile(DoubleImage &img, char* fPath) {
	DoubleImage sec;
	sec.initFrom(img);
//	int maxHeightWidth = img.height > img.width ? img.height : img.width;
    double maxDisp = 200.0;
	sec.scaleBy(255.0 / 200.0);

	int y;

	FILE *fp = fopen(fPath, "wb");
	if (!fp)
		abort();

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png)
		abort();

	png_infop info = png_create_info_struct(png);
	if (!info)
		abort();

	if (setjmp(png_jmpbuf(png)))
		abort();

	png_init_io(png, fp);

	// Output is 8bit depth, RGBA format.
	png_set_IHDR(png, info, sec.width, sec.height, 8,
	PNG_COLOR_TYPE_RGBA,
	PNG_INTERLACE_NONE,
	PNG_COMPRESSION_TYPE_DEFAULT,
	PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png, info);

	// To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
	// Use png_set_filler().
	//png_set_filler(png, 0, PNG_FILLER_AFTER);

	row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * sec.height);
	for (int y = 0; y < sec.height; y++) {
		row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png, info));
	}

	for (int i = 0; i < sec.height; i++) {
	    for (int j = 0; j < sec.width; j++) {
	        int brightness = (int)(sec.img[i][j] + 0.5);
	        if (brightness > 255)
	            brightness = 255;
	        unsigned char col = (unsigned char)(brightness);
	        row_pointers[i][4 * j] = col;
	        row_pointers[i][4 * j + 1] = col;
	        row_pointers[i][4 * j + 2] = col;
	        row_pointers[i][4 * j + 3] = ((unsigned char) 255);
	    }
	}

	png_write_image(png, row_pointers);
	png_write_end(png, NULL);

	for (int y = 0; y < sec.height; y++) {
		free(row_pointers[y]);
	}
	free(row_pointers);

	fclose(fp);
}