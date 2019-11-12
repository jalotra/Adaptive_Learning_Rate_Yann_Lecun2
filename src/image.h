#ifndef IMAGE_H
#define IMAGE_H

typedef struct{
    int h,w,c;
    float *data;
} image;


// Basic operations
float get_pixel(image im, int x, int y, int c);
void set_pixel(image im, int x, int y, int c, float v);
image copy_image(image im);
image get_channel(image im, int c);


// Loading and saving
image make_image(int w, int h, int c);
image load_image(char *filename);
void save_image(image im, const char *name);
void free_image(image im);

#endif