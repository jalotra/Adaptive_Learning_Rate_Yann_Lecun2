### Datasets to play with 
1. MNIST DATASET

To run your model you'll need the dataset. The training images can be found [here](https://pjreddie.com/media/files/mnist_train.tar.gz) and the test images are [here](https://pjreddie.com/media/files/mnist_test.tar.gz), I've preprocessed them for you into PNG format. To get the data you can run:


    wget https://pjreddie.com/media/files/mnist_train.tar.gz
    wget https://pjreddie.com/media/files/mnist_test.tar.gz
    tar xzf mnist_train.tar.gz
    tar xzf mnist_test.tar.gz

We'll also need a list of the images in our training and test set. To do this you can run:

    find train -name \*.png > mnist.train
    find test -name \*.png > mnist.test  
