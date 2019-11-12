#include <assert.h>
#include <stdio.h>
#include <math.h>


//Header FIle for the Algorithm
#include "main_algo.h"
// Header files for matrix methods, NeuralNetwork
#include "../src/matrix.h"
#include "../src/classifier.h"
// #include "../src/data.c"

// Now that I have created the defintion of a Neural Network.
// The main task remains and that is to store the gradeints G1 AND G2.
// HOW is it to be done?
// G1 is calculated in this way:
// FORWARD PROP THROUGH THE MODEL AND BACKPROPOGATE IN THE REVERSE DIRECTION
// CALCULATE THE DERIVATIVE WRT WEIGHTS AND MAKE THIS MATRIX AS G1

// How to calculate the matrix G2?
// 1. UPDATE THE WEIGHTS OF EVERY LAYER BY SOME PSI1 
// 2. THEN FORWARD PROPOGATE THROUGH THE MODEL 
// 3. ALSO BACKPROPOGATE BACKWARDS 
// 4. SAVE THE GRADIENT MATRIX CALCULATED THIS TIME IN A NEW MATRIX G2
// THE LAST TASK STILL REMAINS AND THAT IS TO MAKE SURE THAT AFTER THIS WE CHANGE THE WEIGHTS 
// TO THEIR ORIGINAL VALUE.    

const double const1_alpha = 0.01;
const double const2_gamma = 0.01;

// Calculates the G1 matrix
void calculate_gradient1(int batchsize, model M, data b) 
{
	// data b = random_batch(d, batchsize);
	// Forward Propogate through the model
	matrix output = forward_model(M, b.X);
	// free_matrix(output);
	// Lets calculate the loss at the last layer

	matrix delta = Last_Layer_Loss_Cross_Entropy(b, output); // partial derivative of loss dL/dy
	// Now backpropogate Backwards
	backward_model(M, delta);

	// Now Backward model in itself calls the backward_layer
	// FUnction that saves the derivative wrt weights in dw 
	// Matrix 
	// My job is just to copy the dw matrix in the matrix G1
	for(int i = 0; i < M.n; i++)
	{
		// Free the matrix G1 first
		free_matrix((M.layers+i)->G1);
		(M.layers[i].G1) = copy_matrix((M.layers[i].dw));
		// (M.layers+i)->G1 = scale_matrix(const1_alpha, normalise_psi((M.layers+i)));
	}

	// Free the output matrxix
	// free_matrix(output);
	// free_matrix(delta);
	free_data(b);
	free_matrix(delta);

	return;

}

//calculates the G2 matrix
void calculate_gradient2(int batchsize, model M, data b, matrix psi)
{
	// Now the task is first to update the weights 
	// Of all the layers by some psi
	for(int i = 0; i < M.n; i++)
	{
		// Save curent weights in l->v for use later
		free_matrix((M.layers+i)->v);
		// printf("%d")
		M.layers[i].v = copy_matrix(M.layers[i].w);

		// FIrst free the matrix l->w
		free_matrix(M.layers[i].w);


		M.layers[i].w = scale_matrix(const1_alpha, normalise_psi(psi));
		// free_matrix(new_psi);
	}
	
	// Now Forward Propogate 
	matrix output = forward_model(M, b.X);

	// Calculate  the loss at the last layer
	matrix delta = Last_Layer_Loss_Cross_Entropy(b, output); // partial derivative of loss dL/dy

	// BackPropogate the Loss
	backward_model(M, delta);

	// Now the dw contains the matrix G2
	for(int i = 0; i < M.n; i++)
	{
		// Free the matrix G2
		free_matrix(M.layers[i].G2);
		M.layers[i].G2 = copy_matrix(M.layers[i].dw);
	}

	// Change the current weights with past weights
	for(int i = 0; i < M.n; i++)
	{
		free_matrix(M.layers[i].w);
		M.layers[i].w = copy_matrix(M.layers[i].v);
	}
	// l->w = l->v;

	// FRee the output matrix
	// free_matrix(output);
	free_matrix(delta);
	free_data(b);

	return;
}

