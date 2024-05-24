import logging
import models
import numpy as np
from numpy import argmax
import os
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import time
import utils

# Set Tensorflow logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def main():

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Print dataset information
    utils.print_dataset_information(x_train, y_train, x_test, y_test)

    # Prepare training and test data
    utils.print_heading("Prepare data")
    x_train, y_train, x_test, y_test = _prepare_data(x_train, y_train, x_test, y_test)

    # Evaluate models

    # Evaluation parameters
    batch_size_ = 128
    epochs_ = 5
    learning_rate = 0.01
    n_folds = 3
    test_image_path = os.path.join("../data", "test_image.png")
    verbose_ = 1

    # Array of models to evaluate
    utils.print_heading("Evaluate models")
    models_to_evaluate = [models.get_model1, models.get_model2]

    i = 1
    for get_model_func in models_to_evaluate:
        start_time = time.time()
        utils.print_heading(f'Evaluate model {str(i)}')
        _evaluate_model(x_train, y_train, get_model_func,
            _get_adam_optimizer, batch_size_, epochs_, learning_rate, n_folds, verbose_)
        i += 1
        print("Evaluation took: %s seconds" % (time.time() - start_time))

    # Test a model
    utils.print_heading("Test model 2")
    _test_model(x_train, y_train, x_test, y_test, models.get_model2,
        _get_adam_optimizer, batch_size_, epochs_, learning_rate, test_image_path, verbose_)

def _convert_to_one_hot_encoded(np_array):
    """
    Convert labels to one-hot encoded format

    Parameters
    ----------
    data : numpy array
        Labels
    """
    print(f'shape before: {np_array.shape}')
    print(f'first label before: {np_array[0]}')    
 
    np_array = to_categorical(np_array, num_classes = 10)

    print(f'shape after: {np_array.shape}')
    print(f'first label after: {np_array[0]}')    

    return np_array

def _evaluate_model(x_train, y_train, get_model_func, get_optimizer_func, batch_size_, epochs_, learning_rate_, n_folds, verbose_):
    """
    Evaluate a model using K-fold cross validation

    Parameters
    ----------
    x_train : numpy array
        Training data
    y_train : numpy array
        Training labels
    get_model_func : function
        Function to get the model
    get_optimizer_func : function
        Function to get the optimizer
    batch_size_ : int
        Batch size
    epochs_ : int
        Number of epochs for the model
    learning_rate_ : float
        Learning rate for the optimizer
    n_folds : int
        Number of folds for the cross validation
    verbose_: int
        Verbosity for fitting and evaluating the model
    """
    print(f'Running {n_folds} validation rounds for {epochs_} epochs')

    histories, scores  = list(), list()

    kfold = KFold(n_folds, shuffle=True, random_state=1)

    i = 1

    for train_idx, test_idx in kfold.split(x_train):

        print(f'\nValidation round {str(i)}:\n')

        x_train_, y_train_, x_test, y_test = x_train[train_idx], y_train[train_idx], x_train[test_idx], y_train[test_idx]

        # Get model
        model = get_model_func(get_optimizer_func(learning_rate_))

        # Fit the model
        history = model.fit(x_train_, y_train_, epochs = epochs_, batch_size = batch_size_, validation_data = (x_test, y_test), verbose = verbose_)

        # Evaluate
        validation_loss, validation_accuracy = model.evaluate(x_test, y_test, verbose = verbose_)
        
        print('\nLoss on the validation round: ', validation_loss)
        print('Accuracy on the validation round: ', validation_accuracy)

        histories.append(history)
        scores.append(validation_accuracy)

        i += 1

    # Plot the evaluation results
    utils.plot_learning_curve(histories)
    utils.plot_evaluation_summary(scores)

def _get_adam_optimizer(learning_rate_):
    """
    Get the Adam optimizer
    
    Parameters
    ----------
    learning_rate_ : float
        Learning rate for the optimizer

    """
    return Adam(learning_rate = learning_rate_)

def _load_test_image(filename):
    """
    Load test image

    Parameters
    ----------
    filename : str
        File name of the test image
    """
    img = load_img(filename, color_mode = 'grayscale', target_size = (28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)

    return _normalize(img)

def _normalize(data):
    """
    Normalize data

    Parameters
    ----------
    data : numpy array
        Data to be normalized
    """
    return data.astype('float32') / 255.0

def _prepare_data(x_train, y_train, x_test, y_test):
    """
    Prepare the data for training and testing

    Parameters
    ----------
    x_train: numpy array
        Training data
    y_train: numpy array
        Training labels
    x_test: numpy array
        Test data
    y_test: numpy array
        Test labels     
    """
    # Reshape dataframes to have color channel 
    # needed by Keras (1 = grey, 3 = RGB)
    print("Reshape training samples:")
    x_train = _reshape_data(x_train)

    print("\nReshape test samples:")
    x_test = _reshape_data(x_test)

    # Convert labels to one-hot encoded format
    print("\nConvert training labels to one-hot encoded format:")
    y_train = _convert_to_one_hot_encoded(y_train)

    print("\nConvert test labels to one-hot encoded format:")
    y_test = _convert_to_one_hot_encoded(y_test)

    # Normalize data
    print("\nNormalize training samples")
    x_train = _normalize(x_train)

    print("\nNormalize test samples")
    x_test = _normalize(x_test)

    return x_train, y_train, x_test, y_test

def _reshape_data(np_array):
    """
    Reshape data to have color channel

    Parameters
    ----------
    np_array : numpy array
        Data to be reshaped
    """
    print(f'shape before: {np_array.shape}')   

    np_array = np_array.reshape(np_array.shape[0], 28, 28, 1)

    print(f'shape after: {np_array.shape}')   

    return np_array

def _test_model(x_train, y_train, x_test, y_test, get_model_func, get_optimizer_func, batch_size_, epochs_, learning_rate_, test_image_path, verbose_):
    """
    Evaluate a model using K-fold cross validation

    Parameters
    ----------
    x_train : numpy array
        Training data
    y_train : numpy array
        Training labels
    get_model_func : function
        Function to get the model
    get_optimizer_func : function
        Function to get the optimizer
    batch_size_ : int
        Batch size
    do_softmax : bool
        Whether to do softmax for the predictions (in case of linear ouput layer)
    epochs_ : int
        Number of epochs for the model
    learning_rate_ : float
        Learning rate for the optimizer
    n_folds : int
        Number of folds for the cross validation
    test_image : str
        Filename of the test image
    verbose_: int
        Verbosity for fitting the model
    """
    # Get model to be trained
    model_to_train = get_model_func(get_optimizer_func(learning_rate_))

    # Fit the model with the training data
    model_to_train.fit(x_train, y_train, epochs = epochs_, batch_size = batch_size_, validation_data = (x_test, y_test), verbose = verbose_)
    
    # Save the model
    model_to_train.save('test_model.h5')

    # Load the model
    model = load_model('test_model.h5')

    # Model summary
    print("\nModel summary:")
    print(model.summary())
    
    # Evaluate
    validation_loss, validation_accuracy = model.evaluate(x_test, y_test, verbose = verbose_)
    print("\nTest accuracy and loss:")
    print('\nLoss: ', validation_loss)
    print('Accuracy: ', validation_accuracy)

    # Load and show the test image
    print("\nTest image:")
    test_image = _load_test_image(test_image_path)
    utils.plot_image(test_image.reshape((28, 28)))

    # Predict
    predictions =  model.predict(test_image)
    print("\nPredictions:")
    print(predictions)

    # Show the predicted digit
    print("\nThe predicted digit is: ", argmax(predictions)) 
    
if __name__ == '__main__':
    main()