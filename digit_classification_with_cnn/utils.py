import matplotlib.pyplot as plt
from numpy import mean, std
from seaborn import heatmap
import tensorflow as tf
from tensorflow.math import confusion_matrix   

"""
Utility functions
"""

# Plot figure size
FIGURE_SIZE = (10, 5)

def plot_confusion_matrix(y_actual, y_predictions):
    """
    Plot confusion matrix

    Parameters
    ----------
     y_actual: array like 
        Actual labels
    
    y_predictions: array like 
        Predicted labels
    """
    fig, axes = plt.subplots(figsize = FIGURE_SIZE)

    heatmap(
        confusion_matrix(y_actual, y_predictions),
        annot = True,
        ax = axes,
        square = True
    )

    plt.show()

def plot_evaluation_summary(scores):
    """
    Plot summary of model performance

    Parameters
    ----------
 
    histories: scores
        List of model scores 
    """  
    plt.boxplot(scores)
    plt.show()

    print('Accuracy: Mean = %.5f Standard Deviation = %.5f' % (mean(scores) * 100, std(scores) * 100))

def plot_image(image):
    """
    Plot image

    Parameters
    ----------
 
    image: array like 
        Image data frames
    """  
    plt.imshow(image, cmap='gray')
    plt.show()

def plot_images(images, labels, start, end):
    """
    Plot images

    Parameters
    ----------
    images: array like 
        Array of image data frames
    
    labels: array like 
        Array of image labels
    
    start: int
        Start index
    
    end: int
        End index
    """
    plt.figure(figsize = FIGURE_SIZE)

    for i in range(start, end):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title('Label: {}'.format(labels[i]))

    plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
    plt.show()

def plot_label_distribution(labels):
    """
    Plot histogram of label distribution

    Parameters
    ----------
 
    labels: array like 
        Array of labels
    """    
    plt.figure(figsize = FIGURE_SIZE)
    plt.hist(labels, align = 'left', bins = range(11), rwidth = 0.9)
    plt.title('Distribution')
    plt.xlabel('Label')
    plt.xticks(range(10))
    plt.ylabel('Count')
    plt.show()

def plot_model(model):
    """
    Plot model architecture

    Parameters
    ----------
 
    model: model
        Model to plot
    """      
    tf.keras.utils.plot_model(
        model, 
        show_shapes = True, 
        show_layer_names = True
    )

def plot_learning_curve(histories):
    """
    Plot learning curve

    Parameters
    ----------
 
    histories: list 
        List of model histories
    """
    plt.figure(figsize = FIGURE_SIZE)

    for i in range(len(histories)):
        plt.subplot(2, 1, 1)
        plt.plot(histories[i].history['loss'], color = 'green', label = 'training set')
        plt.plot(histories[i].history['val_loss'], color = 'orange', label = 'validation set')
        plt.title('Losses')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
 
        plt.subplot(2, 1, 2)
        plt.plot(histories[i].history['accuracy'], color='green', label='training set')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='validation set')
        plt.title('Accuracies')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")        

    plt.tight_layout()
    plt.show()
  
def print_dataset_information(x_train, y_train, x_test, y_test):
    """
    Print dataset information

    Parameters
    ----------

    x_train: array like 
        Array of training examples
    
    y_train: array like 
        Array of training labels
    
    x_test: array like 
        Array of test examples
    
    y_test: array like 
        Array of test labels
    """  
    print_heading('Dataset Stats')
    print_dataset_stats(x_train, y_train, x_test, y_test)

    print_heading('Training Set Labels Distribution') 
    plot_label_distribution(y_train)

    print_heading('Test Set Labels Distribution') 
    plot_label_distribution(y_test)

    print_heading("Training Set mages from %s to %s" % (0, 4))
    plot_images(x_train, y_train, 0, 4)

def print_heading(heading):
    """
    Print formatted heading

    Parameters
    ----------
    
    heading: str
        Heading text to print
    """  
    border = '=' * len(heading)

    print(f'\n{border}')
    print(f'{heading}')
    print(f'{border}\n')

def print_dataset_stats(x_train, y_train, x_test, y_test):
    """
    Print dataset stats

    Parameters
    ----------
    
    x_train: array like 
        Array of training examples
    
    y_train: array like 
        Array of training labels
    
    x_test: array like 
        Array of test examples
    
    y_test: array like 
        Array of test labels
    """  
    print('Type of training examples: ', type(x_train))
    print('Shape of training examples: ', x_train.shape)
    print('Number of training examples: ', x_train.shape[0])
    print('Size of an example:', x_train.shape[1:])
    print('Type of training labels: ', type(y_train))
    print('Shape of training labels: ', y_train.shape)
    print('First 5 labels: ' , y_train[0:5:])
    print('Number of test examples: ', x_test.shape[0])