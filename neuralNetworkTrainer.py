from uwimg import *
import time
import sys

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    # print (inputs, outputs)
    l = [make_layer(inputs, 32, RELU),
    		make_layer(32, outputs ,RELU)]
		#make_layer(16, 8, RELU),
            # make_layer(16, outputs, SOFTMAX)]
    return make_model(l)


def iterate_over(model, iters, batchsize, data):
	# Create a model
	
	for iter in range(iters):
		#  Calculate G1 works
		b = random_batch(data, batchsize)
		
		# print_matrix(model.layers[0].G1)
		# time.sleep(1)
		# # print_matrix(Model.G1)
		psi = create_psi(b.X.cols, b.y.cols, 0.1)
		# print("ITER {} ".format(iter))
		calculate_G2(batchsize, model, b, psi)

		# calculate_G1(batchsize, model, b)
		# print_matrix(model.layers[0].G2)
		# # Now calculate the running average of Psi for all the layers in the model
		norm1 = 0
		# for i in range(model.n):
		# 	psi = running_average(psi, (model.layers[i]).G1, (model.layers[i]).G2)
		# 	norm1 = calculate_delta_norm(norm1 ,psi, i)
		# 	print(norm1)

def neural_on_mnist_dataset():
	train_file_path = "mnist.train"
	labels_path = "mnist/mnist.labels"
	test_file_path = "mnist.test"

	# train = load_classification_data(c_char_p(train_file_path.encode('utf-8')), c_char_p(labels_path.encode('utf-8')), 1)
	test  = load_classification_data(c_char_p(test_file_path.encode('utf-8')),c_char_p(labels_path.encode('utf-8')) , 1)
	m = softmax_model(test.X.cols, test.y.cols)
	# Lets print teh G1 matrix 
	# print_matrix(m.layers[0].G1)

	# print_matrix(m.layers[0].w)
	# # free_matrix(m.layers[0].w)
	# # m.layers[0].w = create_psi(m.layers[0].w.rows, m.layers[0].w.cols, 0)
	# print_matrix(m.layers[0].w)

	
	iterate_over(m, 1000, 128, test)
	# Print after the process
	# print_matrix(m.layers[0].G1)
	# print_matrix(m.layers[0].G2)

	# Lets print all the things that model m has
	# print(m)
	# for i in range(m.n ):
	# 	# print_matrix(m.layers[i].in)
	# 	print_matrix(m.layers[i].w)
	# 	print_matrix(m.layers[i].dw)
	# 	print_matrix(m.layers[i].v)
	# 	print_matrix(m.layers[i].out)
	# 	print_matrix(m.layers[i].G1)
	# 	print_matrix(m.layers[i].G2)

	# Model WOrks good
	# Lets check out the gradients function


if __name__ == "__main__":
	neural_on_mnist_dataset()
	
