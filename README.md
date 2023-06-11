# Fish Image Detection using Convolutional Neural Network with Kernels

This project aims to detect fish images using a Convolutional Neural Network (CNN) with kernels. The code provided demonstrates the implementation of the CNN model for this purpose.

Requirements
To run this code, you need the following dependencies:

Python 2 or 3
NumPy
Pillow (PIL)
Dataset
The code uses a dataset of fish and non-fish images. The fish images are located in the "n01440764" folder, and the non-fish images are located in the "DIP3E_Original_Images_CH01" folder. Ensure that the dataset is properly organized and contains the required image files.

Usage
To run the code, follow these steps:

Make sure you have the required dependencies installed.
Adjust the parameters in the code, such as the number of nodes in each layer, learning rate, and number of epochs, as needed.
Execute the main() function to train the model and test its performance on the dataset.
Description
The code begins by importing the necessary libraries and defining the parameters for the CNN model. The preProcess() function loads and preprocesses the fish and non-fish images, shuffling them for better training.

The forwardPropagation() function implements the forward propagation step of the CNN model. It takes an input image, applies a convolution operation using a kernel, passes the result through a hidden layer, and produces an output prediction.

The backwardPropagation() function performs the backward propagation step of the CNN model. It calculates the errors in the output and hidden layers, updates the weights and biases accordingly, and returns the updated parameters.

The imageConvolution() function is used for performing convolution on an input image using a specified kernel. It applies the kernel to small windows of the image and computes the convolved output.

The main() function trains the model by iterating over the dataset for the specified number of epochs. It uses the forward and backward propagation functions to update the weights and biases of the model.

Finally, the model is tested on a separate set of images, and the accuracy is calculated and printed.

Additional Code
The provided code also includes a separate section at the end for testing the image_convolution() function. It demonstrates how the function can be used to apply convolution to an image using a kernel. This section is not directly related to the main purpose of the project but can be used for general image processing tasks.

Please ensure that your dataset is properly organized and that the necessary files and folders are present in the specified locations to avoid any errors while running the code.

Note: It is advisable to refactor and optimize the code as per your specific requirements and standards before deploying it in a production environment.

Result - Achieved an accuracy of 80%
