#ifndef MAIN_ALGO_H
#define MAIN_ALGO_H
#include "../src/matrix.h"
#include "../src/classifier.h"

// Lets define all the function signatures present
extern const double const1_alpha;
extern const double const2_gamma ;

// FIRST FOR learning_rate_finder.c
double calculate_norm(matrix);
matrix create_psi(int, int, double);
matrix normalise_psi(matrix);
matrix running_average(matrix, matrix, matrix);
int calculate_delta_norm(double ,matrix, int layer_number);

// NOW FOR calculate_gradients.c
void calculate_gradient1(int, model, data);
void calculate_gradient2(int batchsize, model, data, matrix);



#endif 