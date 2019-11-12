#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include "matrix.h"

// Some data structures needed by a nueral net

typedef enum 
{
	LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX
}ACTIVATION;

typedef struct
{
	matrix in;    // Take in inputs
	matrix w;     // Current weights
	matrix dw;    // Current weight updates 
	matrix v;     // Past weight updates
	matrix out;   // Output matrix
	matrix G1;
	matrix G2;
	ACTIVATION activation;  // Activation function used by this layer
}layer;


// Represents the data coming in and going out
typedef struct{
	matrix X;         // X represents the input to any layer
	matrix y;         // y represents the inference of any layer
}data;


typedef struct
{
	layer *layers;       // Pointer to layers struct
	int n;               // Number of layers

}model;

// Methods regarding taking input to a model etc;
data load_classification_data(char *images, char *label_file, int bias);
void free_data(data d);
data random_batch(data d, int n);
char *fgetl(FILE *fp); 


// Lets declare all the methods 
void activate_matrix(matrix x, ACTIVATION a);
void gradient_matrix(matrix m, ACTIVATION a, matrix delta);
matrix  forward_layer(layer *l, matrix in);
matrix backward_layer(layer *l, matrix delta);
layer make_layer(int input, int output, ACTIVATION activation);
matrix forward_model(model M, matrix X);
void backward_model(model M, matrix delta);
double cross_entropy_loss(matrix y, matrix p);
matrix Last_Layer_Loss_Cross_Entropy(data b, matrix p);


#endif