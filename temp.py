# Fish Image detection using Convolutional Neural Network with Kernels



from PIL import Image
import numpy as np
import os
from sklearn.utils import shuffle


# Using https://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm for random images
# rows = 480
# columns = 640
# index 0 means fingerprint
# index 1 means not a fingerprint

number_of_input_nodes = 480 * 640
number_of_conv_nodes = 478 * 638
number_of_output_nodes = 2
number_of_hidden_nodes = 50

epochs = 1

# learning rate
lr = 0.01

total_dataset = 173


def main():

    # initializing matrices to store data
    L1weights = np.random.randn(number_of_conv_nodes, number_of_hidden_nodes)
    L2weights = np.random.randn(number_of_hidden_nodes, number_of_output_nodes)

    L1biases = np.random.randn(1, number_of_hidden_nodes)
    L2biases = np.random.randn(1, number_of_output_nodes)

    kernel = np.random.randn(3, 3)

    x_train, y_train, file_names = preProcess()

    x_train = x_train.reshape(total_dataset, 480 * 640, 1) / 255
    y_train = y_train.reshape(total_dataset, 1, 1)
    file_names = file_names.reshape(total_dataset, 1, 1)

    # training model
    for j in range(epochs):
        for i in range(0, 130):
            print i
            prediction, hidden_layer, convolution = forwardPropagation(x_train[i], L1weights, L2weights, L1biases,
                                                                       L2biases, kernel)

            X = x_train[i]
            output = np.array([0, 0])
            output[int(y_train[i])] = 1

            L1weights, L2weights, L1biases, L2biases, kernel = backwardPropagation(X, output, prediction,
                                                                                     L1weights, L2weights, L1biases,
                                                                                     L2biases,
                                                                                     hidden_layer, convolution,
                                                                                     kernel)

    # testing
    total = 0.0
    sum = 0.0

    for i in range(130, 173):
        total += 1
        prediction = np.argmax(
            forwardPropagation(x_train[i], L1weights, L2weights, L1biases, L2biases, kernel)[0]).reshape(1, 1)
        if prediction == y_train[i]:
            sum += 1
        else:
            print file_names[i][0]

        print str(prediction[0]), str(y_train[i][0])

    print str((sum * 100) / total) + "%"

# Randomizes the dataset for better training
def arrayShuffle(X, y, z):
    X, y, z = shuffle(X, y, z, random_state=3)
    return X, y, z


def preProcess():
    arrayofpictures = []
    arrayoflabels = []
    arrayofnames = []

    # processes all the fish images
    folder_dir = "n01440764"

    for images in os.listdir(folder_dir):

        if images.endswith(".JPEG"):
            file_path = "n01440764/" + str(images)
            photo = Image.open(file_path)
            photo = photo.resize((480, 640))
            photo = photo.convert("L")
            data = np.array(photo)
            data = data.reshape(480 * 640)
            arrayofpictures.append(data)
            arrayoflabels.append(1)
            arrayofnames.append(file_path)

    # processes all the non fish images
    folder_dir = "DIP3E_Original_Images_CH01"

    for images in os.listdir(folder_dir):

        if images.endswith(".tif"):
            file_path = folder_dir + "/" + str(images)
            photo = Image.open(file_path)
            photo = photo.resize((480, 640))
            photo = photo.convert("L")
            data = np.array(photo)
            data = data.reshape(480 * 640)
            arrayofpictures.append(data)
            arrayoflabels.append(1)
            arrayofnames.append(file_path)

    return arrayShuffle(np.asarray(arrayofpictures), np.asarray(arrayoflabels), np.asarray(arrayofnames))


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def forwardPropagation(X, W1, W2, B1, B2, kernel):
    convolution = sigmoid(imageConvolution(X, kernel))  # convolutional layer shape is (1, 304964)

    hidden_layer = sigmoid(np.dot(convolution, W1) + B1)  # hidden layer shape is (1, 304964)

    output = sigmoid(np.dot(hidden_layer, W2) + B2)  # output layer shape is (1, 304964)

    return output.reshape(number_of_output_nodes), hidden_layer, convolution


def backwardPropagation(X, Y, output, W1, W2, B1, B2, hidden_layer, convolution, kernel):

    output_delta = (Y - output) * output * (1 - output)  # error in (output layer) * derivative of the sigmoid
    output_delta = output_delta.reshape(1, number_of_output_nodes)

    hidden_layer_error = output_delta.dot(W2.T)  # error in hidden layer
    hidden_layer_error_delta = hidden_layer_error * hidden_layer * (1 - hidden_layer)  # error in (hidden layer) *
    # derivative of the sigmoid

    convolution_error = hidden_layer_error.dot(W1.T)
    convolution_error_delta = convolution_error * convolution * (1 - convolution)
    print np.shape(convolution_error) , np.shape(convolution_error_delta)

    # kernel += X.
    W1 += convolution.T.dot(hidden_layer_error_delta)
    W2 += hidden_layer.T.dot(output_delta)

    B1 += hidden_layer_error_delta
    B2 += output_delta

    return W1, W2, B1, B2, kernel

# Creates Convolution
def imageConvolution(matrix, kernel):
    matrix = matrix.reshape(480, 640)

    k_size = len(kernel)
    m_height, m_width = matrix.shape
    padded = np.pad(matrix, (k_size - 1, k_size - 1), 'minimum')

    output = []
    for i in range(m_height - (k_size - 1)):
        for j in range(m_width - (k_size - 1)):
            output.append(np.sum(matrix[i:k_size + i, j:k_size + j] * kernel))

    output = np.array(output)

    return output.reshape(1, 304964)


main()
