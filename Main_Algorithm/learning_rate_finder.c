#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

// CUSTOM HEADER FILES
//Header FIle for the Algorithm
#include "main_algo.h"
// Header files for matrix methods, NeuralNetwork
#include "../src/matrix.h"
#include "../src/classifier.h"

#define loop_for(i, n) for(int i = 0; i < n ; i++)


// LETS START WITH THE ALGORITHM
// GLOBAL VARIABLES
// DEFINED IN MAIN_ALGO.H

// Returns the L2 norm
double calculate_norm(matrix x)
{
	assert(x.rows >= 1 && x.cols >= 1);
	double ans = 0;  
	loop_for(i, x.rows)
	{
		loop_for(j, x.cols)
		{
			ans += (x.data[i][i])*(x.data[i][j]);
		}
	}

	return sqrt(ans) ;
}

// Creates a Random Vector 
// The reason for this vector to be random so that we dont have a vector which is orthogonal to the hessian matrix
matrix create_psi(int rows, int cols, double s)
{
	// double s = 0.1;
	return random_matrix(rows, cols, s);
}

// Normalise a Matrix Psi
matrix normalise_psi(matrix psi)
{
	// Calculate the norm 
	double current_norm = calculate_norm(psi);

	for(int row = 0; row < psi.rows; ++row)
	{
		for(int col = 0; col < psi.cols; col++)
		{
			psi.data[row][col] /= current_norm;
		}
	}
	// Finally Return the PSI Matrix
	return psi;
}


// Calculates the running average of the vector psi
matrix running_average(matrix psi, matrix gradient1, matrix gradient2)
{
	// Scale with 1 - const2_gamma
	psi = scale_matrix((1.0 - const2_gamma), psi);

	// change g1 to be the differnece in the gradients 
	gradient1 = axpy_matrix(-1, gradient1, gradient2);

	// Scale g1 with const2_gamma/const1_alpha
	// assert(const2_gamma/const1_alpha != nan);
	gradient1 = scale_matrix(const2_gamma/const1_alpha, gradient1);

	// add psi and gradient
	return axpy_matrix(1, psi, gradient1); 

}


// Number of iterations
int calculate_delta_norm(double norm1, matrix psi, int layer_number)
{
	double norm2 = calculate_norm(psi);
	// printf("NORM OF LAYER %d IS %f: \n",layer_number, norm2);

	// IF NORM OF PSI IS LESS THAN 10% ERROR THEN BREAK 
	// WE HAVE GOT THE MAX EIGEN VALUE 
	// printf("LAYER LEARNING RATE: %f", norm1);
	// if((norm2 - norm1) / norm1 < 0.1)
	// {
	// 	printf("MAX EIGEN VALUE REACHED %f", norm1);
	// } 
	norm1 = norm2;


	return norm1;
}




// int main(int argc, char const *argv[])
// {
// 	// Lets run everything that I have wrote here
// 	// FUCK I DIDN'T WROTE UNIT-TESTS
// 	int iters = 500;

// 	void RUN_ALGORITHM(model m, data d, int batch)
// 	{
// 		for(int i = 0; i < iters; i++)
// 		{
// 			data b = random_batch(d, batch);

// 			// Lets calculate the gradients 
// 			// G1
// 			calculate_gradient1(m, b.X, b);

// 			// G2
// 			const s = 0.2;

// 			// Create the matrix psi
// 			matrix psi = create_psi(b.X.rows, b.y.cols, s);

// 			calculate_gradient2(m, b.X, b, psi);

// 			// Update Psi
// 			// Free psi first
// 			free_matrix(psi);


// 			// WRITE FROM HERE 


			
// 			// psi = matrix running_average(psi, matrix gradient1, matrix gradient2)
			


// 			// Lets print the norms for different layers
// 			for(int i = 0; i < m.n; i++)
// 			{
// 				printf("NORM Layer %d IS %f", m.)
// 			}



// 		}
// 	} 


// }

// int main()
// {
// 	matrix psi = create_psi(3, 3, 0.1);
// 	matrix G1 = create_psi(3, 3, 0.2);
// 	matrix G2 = create_psi(3, 3, 0.3);
// 	print_matrix(running_average(psi, G1, G2));
// }