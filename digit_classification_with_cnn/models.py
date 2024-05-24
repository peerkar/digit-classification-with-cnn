from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dropout, BatchNormalization

def get_model1(optimizer_):

    model = Sequential([
        Conv2D(
            activation = 'relu',
            filters = 32,
            kernel_size = (3, 3) 
        ),
        MaxPooling2D(
            pool_size = (2, 2),
       ),
        Flatten(),
        Dense(
            activation = 'relu',
            units = 100
        ),
        Dense(
            activation = 'softmax',
            units = 10
        )
    ])

    model.compile(
        loss = CategoricalCrossentropy(), 
        metrics = ['accuracy'],
        optimizer = optimizer_
    )

    return model

def get_model2(optimizer_):

    model = Sequential([
        Conv2D(
            activation = 'relu',
            filters = 32,
            kernel_initializer=VarianceScaling(),
            kernel_size = (3, 3) 
        ),
        MaxPooling2D(
            pool_size = (2, 2),
            strides = (2, 2)
        ),
        Conv2D(
            activation = 'relu',
            filters = 16,
            kernel_initializer = VarianceScaling(),
            kernel_size = (3, 3) 
        ),
        MaxPooling2D(
            pool_size = (2, 2),
            strides = (2, 2)
        ),
        Flatten(),
        Dense(
            activation = 'relu',
            units = 100
        ),
        Dropout(0.2),
        Dense(
            activation = 'softmax',
            units = 10
        )
    ])

    model.compile(
        loss =  CategoricalCrossentropy(),
        metrics = ['accuracy'],
        optimizer = optimizer_
    )

    return model